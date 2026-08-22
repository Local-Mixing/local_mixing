//! How much of a SOURCE circuit's internal state is a low-degree GF(2) function
//! of a PREDICTOR circuit's wires?
//!
//! A *wire segment* is the interval on a wire between two consecutive writes to
//! it; its value is constant along the interval and is a function of the shared
//! input x. Both circuits are driven by the SAME x on wires 0..blk, zeros
//! elsewhere (the zero slice), so every segment on either side is a function of
//! the same x and the two can be compared directly.
//!
//! For each source segment we test GF(2) span-membership in the predictor set
//!     {1} u {predictor wire values at K cuts}            (degree 1)
//!     ... u {pairwise products of those}                 (degree 2)
//! and, when it is deducible, report the exact combination:
//!   * the region of the source circuit the segment lies in (C / S1 / N / D / S2)
//!   * how many predictor segments the equation needs
//!   * their span = max cut index - min cut index ("how many layers apart")
//!   * whether they all sit on the same predictor wire
//!
//! Degree 2 is a LOWER BOUND unless --pred-wires covers every wire: products
//! among excluded wires are never offered (full degree 2 at 512 wires x K cuts
//! is C(512K,2) regressors — intractable), same caveat as hmap_affine.
//!
//!   segment_deduce --source <c.mpmct1> --pred <p.mpmct1> [--pred-cuts K]
//!                  [--samples N] [--degree 1|2] [--pred-wires W]
//!                  [--source-mode segments|cuts] [--source-cuts K] [--csv out.csv]
use local_mixing::engine::format::read_mpmct;
use local_mixing::circuit::xgate::XGate;
use std::io::Write;
use std::time::Instant;

#[inline]
fn lead(v: &[u64]) -> Option<usize> {
    for (i, &w) in v.iter().enumerate() {
        if w != 0 {
            return Some(i * 64 + w.trailing_zeros() as usize);
        }
    }
    None
}
#[inline]
fn xor_into(dst: &mut [u64], src: &[u64]) {
    for (d, s) in dst.iter_mut().zip(src) {
        *d ^= *s;
    }
}

fn seed_state(num_wires: usize, xs: &[u128], blk: usize) -> Vec<u64> {
    let mut st = vec![0u64; num_wires];
    for (lane, &x) in xs.iter().enumerate() {
        for i in 0..blk {
            if (x >> i) & 1 == 1 {
                st[i] |= 1u64 << lane;
            }
        }
    }
    st
}

// Region of the sandwich a gate belongs to, from its shape + how many N-copies
// have fired. N gate = CNOT  y_i ^= x_i  (target >= n, single positive control
// on target-n). S gates target the first half and read the second half.
fn classify(g: &XGate, n: usize, n_fired: usize) -> &'static str {
    let t = g.target as usize;
    if t >= n {
        if !g.comp && g.ctrls.len() == 1 && g.ctrls[0] == ((t - n) as u16, true) {
            return "N";
        }
        return "other-hi";
    }
    let reads_hi = g.ctrls.iter().any(|&(w, _)| w as usize >= n);
    if reads_hi {
        if n_fired == 0 { "S1" } else { "S2" }
    } else if n_fired == 0 {
        "C"
    } else if n_fired >= n {
        "D"
    } else {
        "C/D-in-N-band"
    }
}

fn main() {
    let a: Vec<String> = std::env::args().collect();
    let get = |k: &str| a.iter().position(|s| s == k).map(|i| a[i + 1].clone());
    let source = get("--source").expect("--source");
    let pred = get("--pred").expect("--pred");
    let pred_cuts: usize = get("--pred-cuts").map(|s| s.parse().unwrap()).unwrap_or(8);
    let samples: usize = get("--samples").map(|s| s.parse().unwrap()).unwrap_or(16384);
    let verify_n: usize = get("--verify").map(|s| s.parse().unwrap()).unwrap_or(4096);
    let degree: usize = get("--degree").map(|s| s.parse().unwrap()).unwrap_or(1);
    let pred_wires: Option<usize> = get("--pred-wires").map(|s| s.parse().unwrap());
    let source_mode = get("--source-mode").unwrap_or_else(|| "segments".into());
    let source_cuts: usize = get("--source-cuts").map(|s| s.parse().unwrap()).unwrap_or(16);
    let blk: usize = get("--blk").map(|s| s.parse().unwrap()).unwrap_or(128);
    let csv_path = get("--csv");
    let tag = get("--tag").unwrap_or_else(|| "run".into());
    assert!(samples % 64 == 0 && verify_n % 64 == 0);

    let t0 = Instant::now();
    let (sg, s_wires) = read_mpmct(&source).expect("read source");
    let (pg, p_wires) = read_mpmct(&pred).expect("read pred");
    let sw = samples / 64;
    eprintln!(
        "source {} gates/{} wires ; pred {} gates/{} wires ; degree {}",
        sg.len(), s_wires, pg.len(), p_wires, degree
    );

    // ---- predictor cut positions & wire subset ----
    let cuts: Vec<usize> =
        (0..=pred_cuts).map(|i| (pg.len() as f64 * i as f64 / pred_cuts as f64).round() as usize).collect();
    let pw_used: Vec<usize> = match pred_wires {
        None => (0..p_wires).collect(),
        Some(w) => {
            // evenly spread across the wire range so all blocks are represented
            (0..w).map(|i| i * p_wires / w).collect()
        }
    };
    let n_lin = cuts.len() * pw_used.len();
    let n_prod = if degree >= 2 { n_lin * (n_lin - 1) / 2 } else { 0 };
    let n_mono = 1 + n_lin + n_prod;
    eprintln!(
        "predictors: {} cuts x {} wires = {} linear, {} products -> {} monomials",
        cuts.len(), pw_used.len(), n_lin, n_prod, n_mono
    );
    assert!(samples > n_mono, "need samples > monomials ({n_mono})");

    // ---- source segments ----
    // segments enumerated as (wire, birth gate index, region); value recorded
    // right after each write, plus the initial value of every wire.
    let mut seg_wire: Vec<usize> = Vec::new();
    let mut seg_birth: Vec<i64> = Vec::new();
    let mut seg_region: Vec<&'static str> = Vec::new();
    let n_half = s_wires / 2;
    if source_mode == "segments" {
        for w in 0..s_wires {
            seg_wire.push(w);
            seg_birth.push(-1);
            seg_region.push("input");
        }
        let mut nf = 0usize;
        for (i, g) in sg.iter().enumerate() {
            let r = classify(g, n_half, nf);
            if r == "N" {
                nf += 1;
            }
            seg_wire.push(g.target as usize);
            seg_birth.push(i as i64);
            seg_region.push(r);
        }
    } else {
        let scut: Vec<usize> =
            (0..=source_cuts).map(|i| (sg.len() as f64 * i as f64 / source_cuts as f64).round() as usize).collect();
        for (ci, c) in scut.iter().enumerate() {
            for w in 0..s_wires {
                seg_wire.push(w);
                seg_birth.push(*c as i64);
                seg_region.push(if ci == 0 { "input" } else { "cut" });
            }
        }
    }
    let n_seg = seg_wire.len();
    eprintln!("source segments: {n_seg} ({source_mode} mode)");

    // ---- sampling ----
    let mut rs = 0x243F6A8885A308D3u64;
    let mut rx = move || -> u128 {
        let mut sm = || -> u64 {
            rs = rs.wrapping_add(0x9E3779B97F4A7C15);
            let mut z = rs;
            z = (z ^ (z >> 30)).wrapping_mul(0xBF58476D1CE4E5B9);
            z = (z ^ (z >> 27)).wrapping_mul(0x94D049BB133111EB);
            z ^ (z >> 31)
        };
        let hi = sm();
        let lo = sm();
        ((hi as u128) << 64) | lo as u128
    };
    let bmask: u128 = if blk >= 128 { u128::MAX } else { (1u128 << blk) - 1 };

    let mut lin_sig = vec![0u64; n_lin * sw];
    let mut seg_sig = vec![0u64; n_seg * sw];

    let mut collect = |wi: usize, xs: &[u128], lin: &mut Vec<u64>, seg: &mut Vec<u64>, sww: usize| {
        // predictor
        let mut st = seed_state(p_wires, xs, blk);
        let mut ci = 0usize;
        for (gi, g) in pg.iter().enumerate() {
            while ci < cuts.len() && cuts[ci] == gi {
                for (j, &w) in pw_used.iter().enumerate() {
                    lin[(ci * pw_used.len() + j) * sww + wi] = st[w];
                }
                ci += 1;
            }
            if ci == cuts.len() {
                break;
            }
            g.apply_lanes(&mut st);
        }
        while ci < cuts.len() {
            for (j, &w) in pw_used.iter().enumerate() {
                lin[(ci * pw_used.len() + j) * sww + wi] = st[w];
            }
            ci += 1;
        }
        // source
        let mut st = seed_state(s_wires, xs, blk);
        if source_mode == "segments" {
            for w in 0..s_wires {
                seg[w * sww + wi] = st[w];
            }
            for (i, g) in sg.iter().enumerate() {
                g.apply_lanes(&mut st);
                seg[(s_wires + i) * sww + wi] = st[g.target as usize];
            }
        } else {
            let scut: Vec<usize> = (0..=source_cuts)
                .map(|i| (sg.len() as f64 * i as f64 / source_cuts as f64).round() as usize)
                .collect();
            let mut ci = 0usize;
            for (gi, g) in sg.iter().enumerate() {
                while ci < scut.len() && scut[ci] == gi {
                    for w in 0..s_wires {
                        seg[(ci * s_wires + w) * sww + wi] = st[w];
                    }
                    ci += 1;
                }
                g.apply_lanes(&mut st);
            }
            while ci < scut.len() {
                for w in 0..s_wires {
                    seg[(ci * s_wires + w) * sww + wi] = st[w];
                }
                ci += 1;
            }
        }
    };

    for wi in 0..sw {
        let xs: Vec<u128> = (0..64).map(|_| rx() & bmask).collect();
        collect(wi, &xs, &mut lin_sig, &mut seg_sig, sw);
    }
    eprintln!("signatures collected ({:.1}s)", t0.elapsed().as_secs_f64());

    // ---- monomial accessor ----
    let prod_pairs: Vec<(usize, usize)> = if degree >= 2 {
        let mut v = Vec::with_capacity(n_prod);
        for i in 0..n_lin {
            for j in (i + 1)..n_lin {
                v.push((i, j));
            }
        }
        v
    } else {
        Vec::new()
    };
    let mono = |idx: usize, out: &mut Vec<u64>| {
        out.clear();
        if idx == 0 {
            out.extend(std::iter::repeat(!0u64).take(sw));
        } else if idx <= n_lin {
            out.extend_from_slice(&lin_sig[(idx - 1) * sw..idx * sw]);
        } else {
            let (i, j) = prod_pairs[idx - n_lin - 1];
            for k in 0..sw {
                out.push(lin_sig[i * sw + k] & lin_sig[j * sw + k]);
            }
        }
    };

    // ---- build predictor basis with tags ----
    let tw = n_mono.div_ceil(64);
    let mut piv: Vec<i32> = vec![-1; samples];
    let mut bsig: Vec<Vec<u64>> = Vec::new();
    let mut btag: Vec<Vec<u64>> = Vec::new();
    let mut buf: Vec<u64> = Vec::with_capacity(sw);
    for idx in 0..n_mono {
        mono(idx, &mut buf);
        let mut v = buf.clone();
        let mut tg = vec![0u64; tw];
        tg[idx / 64] |= 1u64 << (idx % 64);
        loop {
            match lead(&v) {
                None => break,
                Some(p) => {
                    if piv[p] >= 0 {
                        let i = piv[p] as usize;
                        xor_into(&mut v, &bsig[i]);
                        xor_into(&mut tg, &btag[i]);
                    } else {
                        piv[p] = bsig.len() as i32;
                        bsig.push(v);
                        btag.push(tg);
                        break;
                    }
                }
            }
        }
    }
    eprintln!("predictor basis rank {} ({:.1}s)", bsig.len(), t0.elapsed().as_secs_f64());

    // ---- test each source segment for span membership ----
    struct Hit {
        seg: usize,
        n_lin_terms: usize,
        n_prod_terms: usize,
        span: usize,
        same_wire: bool,
        uses_const: bool,
        cut_lo: usize,
        cut_hi: usize,
        pred_c0: i64,          // cut of the first predictor term
        pred_w0: i64,          // wire of the first predictor term
        same_as_src_wire: bool,// some predictor term sits on the SAME wire index as the source segment
    }
    let mut hits: Vec<Hit> = Vec::new();
    for s in 0..n_seg {
        let mut v: Vec<u64> = seg_sig[s * sw..(s + 1) * sw].to_vec();
        let mut tg = vec![0u64; tw];
        let mut ok = true;
        loop {
            match lead(&v) {
                None => break,
                Some(p) => {
                    if piv[p] >= 0 {
                        let i = piv[p] as usize;
                        xor_into(&mut v, &bsig[i]);
                        xor_into(&mut tg, &btag[i]);
                    } else {
                        ok = false;
                        break;
                    }
                }
            }
        }
        if !ok {
            continue;
        }
        // decode the combination
        let mut cuts_used: Vec<usize> = Vec::new();
        let mut wires_used: Vec<usize> = Vec::new();
        let (mut nl, mut np) = (0usize, 0usize);
        let mut uses_const = false;
        for m in 0..n_mono {
            if (tg[m / 64] >> (m % 64)) & 1 == 0 {
                continue;
            }
            if m == 0 {
                uses_const = true;
            } else if m <= n_lin {
                nl += 1;
                let li = m - 1;
                cuts_used.push(li / pw_used.len());
                wires_used.push(pw_used[li % pw_used.len()]);
            } else {
                np += 1;
                let (i, j) = prod_pairs[m - n_lin - 1];
                for li in [i, j] {
                    cuts_used.push(li / pw_used.len());
                    wires_used.push(pw_used[li % pw_used.len()]);
                }
            }
        }
        if nl + np == 0 {
            // segment is a constant on the coset; not a real reconstruction
            continue;
        }
        let span = if cuts_used.is_empty() {
            0
        } else {
            cuts_used.iter().max().unwrap() - cuts_used.iter().min().unwrap()
        };
        let same_wire = wires_used.windows(2).all(|w| w[0] == w[1]);
        let cut_lo = cuts_used.iter().copied().min().unwrap_or(0);
        let cut_hi = cuts_used.iter().copied().max().unwrap_or(0);
        let pred_c0 = cuts_used.first().map(|&c| c as i64).unwrap_or(-1);
        let pred_w0 = wires_used.first().map(|&w| w as i64).unwrap_or(-1);
        let same_as_src_wire = wires_used.iter().any(|&w| w == seg_wire[s]);
        hits.push(Hit { seg: s, n_lin_terms: nl, n_prod_terms: np, span, same_wire, uses_const,
                        cut_lo, cut_hi, pred_c0, pred_w0, same_as_src_wire });
    }
    eprintln!("candidate hits {} ({:.1}s); verifying...", hits.len(), t0.elapsed().as_secs_f64());

    // ---- verify on fresh samples ----
    // Re-derive each hit's combination and check it on independent inputs.
    let vw = verify_n / 64;
    let mut vlin = vec![0u64; n_lin * vw];
    let mut vseg = vec![0u64; n_seg * vw];
    for wi in 0..vw {
        let xs: Vec<u128> = (0..64).map(|_| rx() & bmask).collect();
        collect(wi, &xs, &mut vlin, &mut vseg, vw);
    }
    // recompute tags (we kept only summary stats, so redo the reduction storing tags)
    let mut verified = 0usize;
    let mut failed = 0usize;
    for h in &hits {
        let s = h.seg;
        let mut v: Vec<u64> = seg_sig[s * sw..(s + 1) * sw].to_vec();
        let mut tg = vec![0u64; tw];
        loop {
            match lead(&v) {
                None => break,
                Some(p) => {
                    let i = piv[p] as usize;
                    xor_into(&mut v, &bsig[i]);
                    xor_into(&mut tg, &btag[i]);
                }
            }
        }
        let mut bad = false;
        for k in 0..vw {
            let mut acc = vseg[s * vw + k];
            for m in 0..n_mono {
                if (tg[m / 64] >> (m % 64)) & 1 == 0 {
                    continue;
                }
                acc ^= if m == 0 {
                    !0u64
                } else if m <= n_lin {
                    vlin[(m - 1) * vw + k]
                } else {
                    let (i, j) = prod_pairs[m - n_lin - 1];
                    vlin[i * vw + k] & vlin[j * vw + k]
                };
            }
            if acc != 0 {
                bad = true;
                break;
            }
        }
        if bad {
            failed += 1;
        } else {
            verified += 1;
        }
    }

    // ---- report ----
    let kept: Vec<&Hit> = hits.iter().collect();
    println!("=== {tag} : source={} pred={} degree={} ===", source, pred, degree);
    println!("source segments tested   : {n_seg}");
    println!("deducible (pre-verify)   : {}", hits.len());
    println!("VERIFIED on fresh samples: {verified}   (failed {failed})");
    if n_seg > 0 {
        println!("fraction deducible       : {:.2}%", 100.0 * verified as f64 / n_seg as f64);
    }
    // region breakdown
    use std::collections::BTreeMap;
    let mut by_region: BTreeMap<&str, usize> = BTreeMap::new();
    let mut tot_region: BTreeMap<&str, usize> = BTreeMap::new();
    for s in 0..n_seg {
        *tot_region.entry(seg_region[s]).or_insert(0) += 1;
    }
    for h in &kept {
        *by_region.entry(seg_region[h.seg]).or_insert(0) += 1;
    }
    println!("\nby region of the source circuit:");
    println!("  {:<16} {:>10} {:>10} {:>8}", "region", "deducible", "total", "pct");
    for (r, t) in &tot_region {
        let d = by_region.get(r).copied().unwrap_or(0);
        println!("  {:<16} {:>10} {:>10} {:>7.1}%", r, d, t, 100.0 * d as f64 / *t as f64);
    }
    if !kept.is_empty() {
        let mut nt: Vec<usize> = kept.iter().map(|h| h.n_lin_terms + h.n_prod_terms).collect();
        nt.sort_unstable();
        let mut sp: Vec<usize> = kept.iter().map(|h| h.span).collect();
        sp.sort_unstable();
        let same = kept.iter().filter(|h| h.same_wire).count();
        println!("\nequation shape (over {} deducible segments):", kept.len());
        println!("  predictor terms : min {} median {} max {}", nt[0], nt[nt.len() / 2], nt[nt.len() - 1]);
        println!("  cut span        : min {} median {} max {}  (of {} cuts)", sp[0], sp[sp.len() / 2], sp[sp.len() - 1], cuts.len() - 1);
        println!("  same predictor wire : {} / {} ({:.1}%)", same, kept.len(), 100.0 * same as f64 / kept.len() as f64);
        let withconst = kept.iter().filter(|h| h.uses_const).count();
        println!("  uses constant term  : {}", withconst);
        // WHICH predictor cuts carry the information? boundary cuts (0 = the shared
        // input, last = the shared output) are identical across pipeline stages,
        // so relations resting only on them say nothing about the mixing.
        let ncut = cuts.len();
        let mut cuthist = vec![0usize; ncut];
        for h in kept.iter() {
            for c in h.cut_lo..=h.cut_hi {
                if h.cut_lo == h.cut_hi || c == h.cut_lo || c == h.cut_hi {
                    cuthist[c] += 1;
                }
            }
        }
        println!("  predictor cuts used (cut index -> #equations):");
        for (c, n) in cuthist.iter().enumerate() {
            if *n > 0 {
                let kind = if c == 0 { " [shared INPUT]" } else if c == ncut - 1 { " [shared OUTPUT]" } else { " interior" };
                println!("      cut {:>2} of {:>2}{:<16} : {}", c, ncut - 1, kind, n);
            }
        }
        let interior_only = kept.iter().filter(|h| h.cut_lo > 0 && h.cut_hi < ncut - 1).count();
        println!("  equations using ONLY interior cuts (real mixing leak): {}", interior_only);
        if degree >= 2 {
            let withprod = kept.iter().filter(|h| h.n_prod_terms > 0).count();
            println!("  uses >=1 product    : {} ({:.1}%)", withprod, 100.0 * withprod as f64 / kept.len() as f64);
        }
    }
    if let Some(p) = csv_path {
        let mut f = std::fs::File::create(&p).expect("csv");
        writeln!(f, "tag,degree,seg_wire,seg_birth,src_cut,region,lin_terms,prod_terms,span,same_wire,uses_const,cut_lo,cut_hi,pred_c0,pred_w0,same_as_src_wire").unwrap();
        for h in &kept {
            writeln!(
                f, "{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{}",
                tag, degree, seg_wire[h.seg], seg_birth[h.seg],
                if source_mode == "cuts" { (h.seg / s_wires) as i64 } else { -1 },
                seg_region[h.seg],
                h.n_lin_terms, h.n_prod_terms, h.span, h.same_wire, h.uses_const, h.cut_lo, h.cut_hi,
                h.pred_c0, h.pred_w0, h.same_as_src_wire
            ).unwrap();
        }
        eprintln!("wrote {p}");
    }
    eprintln!("done ({:.1}s)", t0.elapsed().as_secs_f64());
}
