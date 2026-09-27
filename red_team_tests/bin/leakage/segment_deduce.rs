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
//!   * the region of a recognized 256-wire sandwich source (C / S1 / N / D /
//!     S2), or `gate` for another source
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
//!                  [--cut-lo G --cut-hi G]
//!                  [--source-mode segments|cuts] [--source-cuts K]
//!                  [--source-gate-lo G --source-gate-hi G]
//!                  [--csv out.csv] [--eq-dump out.txt] [--eq-region all|REGION]
//!
//! Predictor cuts include both endpoints of the gate window [cut-lo, cut-hi].
//! The default window is the whole predictor. `--eq-dump` writes the complete
//! verified equation for each retained source segment; `--eq-region` optionally
//! restricts that dump to one source-region label and defaults to `all`.
//! `--source-gate-lo/--source-gate-hi` restrict segments mode to post-write
//! segments born at gates in the half-open interval [lo, hi), excluding the
//! source's initial wire segments. The two source-window flags must be paired.
use local_mixing::circuit::xgate::XGate;
use local_mixing::engine::format::read_mpmct;
use rayon::prelude::*;
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

// N gate = CNOT y_i ^= x_i (target >= n, single positive control on target-n).
fn is_neck_gate(g: &XGate, n: usize) -> bool {
    let t = g.target as usize;
    t >= n && !g.comp && g.ctrls.len() == 1 && g.ctrls[0] == ((t - n) as u16, true)
}

// The sandwich's final float moves N gates but never changes the relative order
// of non-N gates. Therefore the first half of the non-N stream is C interleaved
// with S1, and the second is D interleaved with S2. Within either half, S gates
// are exactly the low-target gates that read a high wire.
fn classify_sandwich(
    g: &XGate,
    n: usize,
    non_neck_seen: usize,
    non_neck_split: usize,
) -> &'static str {
    let t = g.target as usize;
    if t >= n {
        if is_neck_gate(g, n) {
            return "N";
        }
        return "other-hi";
    }
    let reads_hi = g.ctrls.iter().any(|&(w, _)| w as usize >= n);
    if non_neck_seen < non_neck_split {
        if reads_hi { "S1" } else { "C" }
    } else if reads_hi {
        "S2"
    } else {
        "D"
    }
}

fn main() {
    let a: Vec<String> = std::env::args().collect();
    let get = |k: &str| a.iter().position(|s| s == k).map(|i| a[i + 1].clone());
    let source = get("--source").expect("--source");
    let pred = get("--pred").expect("--pred");
    let pred_cuts: usize = get("--pred-cuts").map(|s| s.parse().unwrap()).unwrap_or(8);
    let samples: usize = get("--samples")
        .map(|s| s.parse().unwrap())
        .unwrap_or(16384);
    let verify_n: usize = get("--verify").map(|s| s.parse().unwrap()).unwrap_or(4096);
    let degree: usize = get("--degree").map(|s| s.parse().unwrap()).unwrap_or(1);
    let pred_wires: Option<usize> = get("--pred-wires").map(|s| s.parse().unwrap());
    // Force-include specific predictor wires on top of the --pred-wires spread
    // (e.g. band/ancilla wires an adversary can identify syntactically but that
    // an even spread across the wire range would skip).
    let pred_extra_wires: Vec<usize> = get("--pred-extra-wires")
        .map(|s| s.split(',').map(|t| t.trim().parse().unwrap()).collect())
        .unwrap_or_default();
    let source_mode = get("--source-mode").unwrap_or_else(|| "segments".into());
    let source_cuts: usize = get("--source-cuts")
        .map(|s| s.parse().unwrap())
        .unwrap_or(16);
    let source_gate_lo_arg: Option<usize> = get("--source-gate-lo").map(|s| s.parse().unwrap());
    let source_gate_hi_arg: Option<usize> = get("--source-gate-hi").map(|s| s.parse().unwrap());
    let blk: usize = get("--blk").map(|s| s.parse().unwrap()).unwrap_or(128);
    let csv_path = get("--csv");
    let eq_dump_path = get("--eq-dump");
    let eq_region = get("--eq-region").unwrap_or_else(|| "all".into());
    let tag = get("--tag").unwrap_or_else(|| "run".into());
    assert!(pred_cuts > 0, "--pred-cuts must be positive");
    assert!(verify_n > 0, "--verify must be positive");
    assert!(samples % 64 == 0 && verify_n % 64 == 0);

    let t0 = Instant::now();
    let (sg, s_wires) = read_mpmct(&source).expect("read source");
    let (pg, p_wires) = read_mpmct(&pred).expect("read pred");
    let source_gate_window = match (source_gate_lo_arg, source_gate_hi_arg) {
        (None, None) => None,
        (Some(lo), Some(hi)) => {
            assert!(
                source_mode == "segments",
                "--source-gate-lo/--source-gate-hi require --source-mode segments"
            );
            assert!(
                lo <= hi,
                "--source-gate-lo ({lo}) must not exceed --source-gate-hi ({hi})"
            );
            assert!(
                hi <= sg.len(),
                "--source-gate-hi ({hi}) exceeds source length ({})",
                sg.len()
            );
            Some((lo, hi))
        }
        _ => panic!("--source-gate-lo and --source-gate-hi must be used together"),
    };
    let cut_lo_g: usize = get("--cut-lo").map(|s| s.parse().unwrap()).unwrap_or(0);
    let cut_hi_g: usize = get("--cut-hi")
        .map(|s| s.parse().unwrap())
        .unwrap_or(pg.len());
    assert!(
        cut_lo_g <= cut_hi_g,
        "--cut-lo ({cut_lo_g}) must not exceed --cut-hi ({cut_hi_g})"
    );
    assert!(
        cut_hi_g <= pg.len(),
        "--cut-hi ({cut_hi_g}) exceeds predictor length ({})",
        pg.len()
    );
    let sw = samples / 64;
    eprintln!(
        "source {} gates/{} wires ; pred {} gates/{} wires ; degree {}",
        sg.len(),
        s_wires,
        pg.len(),
        p_wires,
        degree
    );
    if let Some((lo, hi)) = source_gate_window {
        eprintln!(
            "source segment window: write-born segments at gates {lo}..{hi} (half-open; initial segments excluded)"
        );
    }

    // ---- predictor cut positions & wire subset ----
    let cuts: Vec<usize> = (0..=pred_cuts)
        .map(|i| {
            (cut_lo_g as f64 + (cut_hi_g as f64 - cut_lo_g as f64) * i as f64 / pred_cuts as f64)
                .round() as usize
        })
        .collect();
    eprintln!(
        "predictor cut window: gates {cut_lo_g}..{cut_hi_g} ({} cuts including endpoints)",
        cuts.len()
    );
    let pw_used: Vec<usize> = {
        let mut v: Vec<usize> = match pred_wires {
            None => (0..p_wires).collect(),
            Some(w) => {
                // evenly spread across the wire range so all blocks are represented
                (0..w).map(|i| i * p_wires / w).collect()
            }
        };
        for &w in &pred_extra_wires {
            assert!(w < p_wires, "--pred-extra-wires {w} >= predictor wires {p_wires}");
            v.push(w);
        }
        v.sort_unstable();
        v.dedup();
        v
    };
    let n_lin = cuts.len() * pw_used.len();
    let n_prod = if degree >= 2 {
        n_lin * (n_lin - 1) / 2
    } else {
        0
    };
    let n_mono = 1 + n_lin + n_prod;
    eprintln!(
        "predictors: {} cuts x {} wires = {} linear, {} products -> {} monomials",
        cuts.len(),
        pw_used.len(),
        n_lin,
        n_prod,
        n_mono
    );
    assert!(samples > n_mono, "need samples > monomials ({n_mono})");

    // ---- source segments ----
    // segments enumerated as (wire, birth gate index, region); value recorded
    // right after each write, plus the initial value of every wire.
    let mut seg_wire: Vec<usize> = Vec::new();
    let mut seg_birth: Vec<i64> = Vec::new();
    let mut seg_region: Vec<&'static str> = Vec::new();
    let n_half = s_wires / 2;
    let neck_gates = if s_wires == 256 {
        sg.iter().filter(|g| is_neck_gate(g, n_half)).count()
    } else {
        0
    };
    let non_neck_gates = sg.len() - neck_gates;
    let sandwich_regions = s_wires == 256 && neck_gates == n_half && non_neck_gates % 2 == 0;
    let non_neck_split = non_neck_gates / 2;
    if sandwich_regions {
        eprintln!(
            "source regions: recognized sandwich ({} non-neck gates per half, {neck_gates} N)",
            non_neck_split
        );
    } else {
        eprintln!("source regions: generic gate labels");
    }
    if source_mode == "segments" {
        if source_gate_window.is_none() {
            for w in 0..s_wires {
                seg_wire.push(w);
                seg_birth.push(-1);
                seg_region.push(if w < blk {
                    "input-active"
                } else {
                    "input-zero"
                });
            }
        }
        let mut non_neck_seen = 0usize;
        for (i, g) in sg.iter().enumerate() {
            let r = if sandwich_regions {
                classify_sandwich(g, n_half, non_neck_seen, non_neck_split)
            } else {
                "gate"
            };
            if r != "N" {
                non_neck_seen += 1;
            }
            if let Some((lo, hi)) = source_gate_window
                && (i < lo || i >= hi)
            {
                continue;
            }
            seg_wire.push(g.target as usize);
            seg_birth.push(i as i64);
            seg_region.push(r);
        }
    } else {
        let scut: Vec<usize> = (0..=source_cuts)
            .map(|i| (sg.len() as f64 * i as f64 / source_cuts as f64).round() as usize)
            .collect();
        for (ci, c) in scut.iter().enumerate() {
            for w in 0..s_wires {
                seg_wire.push(w);
                seg_birth.push(*c as i64);
                seg_region.push(if ci > 0 {
                    "cut"
                } else if w < blk {
                    "input-active"
                } else {
                    "input-zero"
                });
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
    let bmask: u128 = if blk >= 128 {
        u128::MAX
    } else {
        (1u128 << blk) - 1
    };

    let mut lin_sig = vec![0u64; n_lin * sw];
    let mut seg_sig = vec![0u64; n_seg * sw];

    let collect = |wi: usize, xs: &[u128], lin: &mut Vec<u64>, seg: &mut Vec<u64>, sww: usize| {
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
            if let Some((lo, hi)) = source_gate_window {
                for (i, g) in sg.iter().take(hi).enumerate() {
                    g.apply_lanes(&mut st);
                    if i >= lo {
                        seg[(i - lo) * sww + wi] = st[g.target as usize];
                    }
                }
            } else {
                for w in 0..s_wires {
                    seg[w * sww + wi] = st[w];
                }
                for (i, g) in sg.iter().enumerate() {
                    g.apply_lanes(&mut st);
                    seg[(s_wires + i) * sww + wi] = st[g.target as usize];
                }
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
    eprintln!(
        "predictor basis rank {} ({:.1}s)",
        bsig.len(),
        t0.elapsed().as_secs_f64()
    );

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
        pred_c0: i64,           // cut of the first predictor term
        pred_w0: i64,           // wire of the first predictor term
        same_as_src_wire: bool, // some predictor term sits on the SAME wire index as the source segment
        // Selected non-constant monomial ids in the predictor basis. Keeping
        // compact ids avoids carrying a decoded product-sized enum per term.
        terms: Vec<usize>,
        verified: bool,
    }
    // Keep one ordered result slot per segment so parallel execution cannot
    // perturb hit order. Each worker holds only the current segment's scratch.
    let hit_slots: Vec<Option<Hit>> = (0..n_seg)
        .into_par_iter()
        .map_init(
            || (vec![0u64; sw], vec![0u64; tw]),
            |(v, tg), s| {
                // Most source segments are not members. Reject them using only
                // the signature basis; carrying and XORing the additional tag
                // vector is unnecessary for nonmembers.
                v.copy_from_slice(&seg_sig[s * sw..(s + 1) * sw]);
                let mut ok = true;
                loop {
                    match lead(&v) {
                        None => break,
                        Some(p) => {
                            if piv[p] >= 0 {
                                let i = piv[p] as usize;
                                xor_into(v, &bsig[i]);
                            } else {
                                ok = false;
                                break;
                            }
                        }
                    }
                }
                if !ok {
                    return None;
                }

                // Re-run the identical deterministic reduction only for a
                // member, this time carrying tags to recover its equation.
                v.copy_from_slice(&seg_sig[s * sw..(s + 1) * sw]);
                tg.fill(0);
                while let Some(p) = lead(&v) {
                    if piv[p] < 0 {
                        return None;
                    }
                    let i = piv[p] as usize;
                    xor_into(v, &bsig[i]);
                    xor_into(tg, &btag[i]);
                }

                // Decode the combination while this item's tag is still local.
                let mut cuts_used: Vec<usize> = Vec::new();
                let mut wires_used: Vec<usize> = Vec::new();
                let mut terms: Vec<usize> = Vec::new();
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
                        let cut = li / pw_used.len();
                        let wire = pw_used[li % pw_used.len()];
                        cuts_used.push(cut);
                        wires_used.push(wire);
                        terms.push(m);
                    } else {
                        np += 1;
                        let (i, j) = prod_pairs[m - n_lin - 1];
                        let left_cut = i / pw_used.len();
                        let left_wire = pw_used[i % pw_used.len()];
                        let right_cut = j / pw_used.len();
                        let right_wire = pw_used[j % pw_used.len()];
                        cuts_used.extend([left_cut, right_cut]);
                        wires_used.extend([left_wire, right_wire]);
                        terms.push(m);
                    }
                }
                if nl + np == 0 {
                    // Segment is a constant on the coset; not a real reconstruction.
                    return None;
                }
                let span = cuts_used.iter().max().unwrap() - cuts_used.iter().min().unwrap();
                let same_wire = wires_used.windows(2).all(|w| w[0] == w[1]);
                let cut_lo = cuts_used.iter().copied().min().unwrap();
                let cut_hi = cuts_used.iter().copied().max().unwrap();
                let pred_c0 = cuts_used[0] as i64;
                let pred_w0 = wires_used[0] as i64;
                let same_as_src_wire = wires_used.iter().any(|&w| w == seg_wire[s]);
                Some(Hit {
                    seg: s,
                    n_lin_terms: nl,
                    n_prod_terms: np,
                    span,
                    same_wire,
                    uses_const,
                    cut_lo,
                    cut_hi,
                    pred_c0,
                    pred_w0,
                    same_as_src_wire,
                    terms,
                    verified: false,
                })
            },
        )
        .collect();
    let mut hits: Vec<Hit> = hit_slots.into_iter().flatten().collect();
    eprintln!(
        "candidate hits {} ({:.1}s); verifying...",
        hits.len(),
        t0.elapsed().as_secs_f64()
    );

    // ---- verify on fresh samples ----
    // Check each decoded hit equation on independent inputs.
    let vw = verify_n / 64;
    let mut vlin = vec![0u64; n_lin * vw];
    let mut vseg = vec![0u64; n_seg * vw];
    for wi in 0..vw {
        let xs: Vec<u128> = (0..64).map(|_| rx() & bmask).collect();
        collect(wi, &xs, &mut vlin, &mut vseg, vw);
    }
    // Verify the already-decoded equation for each candidate. The candidate
    // equations are independent and all sampled signatures are read-only.
    hits.par_iter_mut().for_each(|h| {
        let s = h.seg;
        h.verified = (0..vw).all(|k| {
            let mut acc = vseg[s * vw + k];
            if h.uses_const {
                acc ^= !0u64;
            }
            for &m in &h.terms {
                acc ^= if m <= n_lin {
                    vlin[(m - 1) * vw + k]
                } else {
                    let (i, j) = prod_pairs[m - n_lin - 1];
                    vlin[i * vw + k] & vlin[j * vw + k]
                };
            }
            acc == 0
        });
    });
    let verified = hits.iter().filter(|h| h.verified).count();
    let failed = hits.len() - verified;

    // ---- report ----
    let kept: Vec<&Hit> = hits.iter().filter(|h| h.verified).collect();
    // Persist result artifacts before the first stdout report write. If a
    // downstream stdout consumer disappears, the expensive run is still saved.
    if let Some(p) = csv_path {
        let mut f = std::fs::File::create(&p).expect("csv");
        writeln!(f, "tag,degree,seg_wire,seg_birth,src_cut,region,lin_terms,prod_terms,span,same_wire,uses_const,cut_lo,cut_hi,pred_c0,pred_w0,same_as_src_wire").unwrap();
        for h in &kept {
            writeln!(
                f,
                "{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{}",
                tag,
                degree,
                seg_wire[h.seg],
                seg_birth[h.seg],
                if source_mode == "cuts" {
                    (h.seg / s_wires) as i64
                } else {
                    -1
                },
                seg_region[h.seg],
                h.n_lin_terms,
                h.n_prod_terms,
                h.span,
                h.same_wire,
                h.uses_const,
                h.cut_lo,
                h.cut_hi,
                h.pred_c0,
                h.pred_w0,
                h.same_as_src_wire
            )
            .unwrap();
        }
        eprintln!("wrote {p}");
    }
    if let Some(p) = eq_dump_path {
        let mut f = std::fs::File::create(&p).expect("eq-dump");
        let mut written = 0usize;
        for h in &kept {
            let region = seg_region[h.seg];
            if eq_region != "all" && eq_region != region {
                continue;
            }
            let terms: Vec<String> = h
                .terms
                .iter()
                .map(|&m| {
                    if m <= n_lin {
                        let li = m - 1;
                        let cut = li / pw_used.len();
                        let wire = pw_used[li % pw_used.len()];
                        format!("w{wire}@g{}", cuts[cut])
                    } else {
                        let (i, j) = prod_pairs[m - n_lin - 1];
                        let left_cut = i / pw_used.len();
                        let left_wire = pw_used[i % pw_used.len()];
                        let right_cut = j / pw_used.len();
                        let right_wire = pw_used[j % pw_used.len()];
                        format!(
                            "(w{left_wire}@g{}*w{right_wire}@g{})",
                            cuts[left_cut], cuts[right_cut]
                        )
                    }
                })
                .collect();
            writeln!(
                f,
                "seg_wire={} birth={} region={} n_lin={} n_prod={} const={} : {}",
                seg_wire[h.seg],
                seg_birth[h.seg],
                region,
                h.n_lin_terms,
                h.n_prod_terms,
                h.uses_const,
                terms.join(" ")
            )
            .unwrap();
            written += 1;
        }
        eprintln!("wrote {written} verified equations to {p} (region={eq_region})");
    }

    println!(
        "=== {tag} : source={} pred={} degree={} ===",
        source, pred, degree
    );
    println!("source segments tested   : {n_seg}");
    println!("deducible (pre-verify)   : {}", hits.len());
    println!("VERIFIED on fresh samples: {verified}   (failed {failed})");
    if n_seg > 0 {
        println!(
            "fraction deducible       : {:.2}%",
            100.0 * verified as f64 / n_seg as f64
        );
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
    println!(
        "  {:<16} {:>10} {:>10} {:>8}",
        "region", "deducible", "total", "pct"
    );
    for (r, t) in &tot_region {
        let d = by_region.get(r).copied().unwrap_or(0);
        println!(
            "  {:<16} {:>10} {:>10} {:>7.1}%",
            r,
            d,
            t,
            100.0 * d as f64 / *t as f64
        );
    }
    if !kept.is_empty() {
        let mut nt: Vec<usize> = kept
            .iter()
            .map(|h| h.n_lin_terms + h.n_prod_terms)
            .collect();
        nt.sort_unstable();
        let mut sp: Vec<usize> = kept.iter().map(|h| h.span).collect();
        sp.sort_unstable();
        let same = kept.iter().filter(|h| h.same_wire).count();
        println!("\nequation shape (over {} deducible segments):", kept.len());
        println!(
            "  predictor terms : min {} median {} max {}",
            nt[0],
            nt[nt.len() / 2],
            nt[nt.len() - 1]
        );
        println!(
            "  cut span        : min {} median {} max {}  (of {} cuts)",
            sp[0],
            sp[sp.len() / 2],
            sp[sp.len() - 1],
            cuts.len() - 1
        );
        println!(
            "  same predictor wire : {} / {} ({:.1}%)",
            same,
            kept.len(),
            100.0 * same as f64 / kept.len() as f64
        );
        let withconst = kept.iter().filter(|h| h.uses_const).count();
        println!("  uses constant term  : {}", withconst);
        // WHICH predictor cuts carry the information? Actual gate 0 and
        // pg.len() are the shared input/output boundaries. The endpoints of a
        // local cut window remain genuine interior states.
        let ncut = cuts.len();
        let mut cuthist = vec![0usize; ncut];
        for h in kept.iter() {
            let mut used = Vec::new();
            for &m in &h.terms {
                if m <= n_lin {
                    used.push((m - 1) / pw_used.len());
                } else {
                    let (i, j) = prod_pairs[m - n_lin - 1];
                    used.extend([i / pw_used.len(), j / pw_used.len()]);
                }
            }
            used.sort_unstable();
            used.dedup();
            for c in used {
                cuthist[c] += 1;
            }
        }
        println!("  predictor cuts used (cut index -> #equations):");
        for (c, n) in cuthist.iter().enumerate() {
            if *n > 0 {
                let kind = if cuts[c] == 0 {
                    " [shared INPUT]"
                } else if cuts[c] == pg.len() {
                    " [shared OUTPUT]"
                } else {
                    " interior"
                };
                println!(
                    "      cut {:>2} of {:>2} @ gate {:>8}{:<16} : {}",
                    c,
                    ncut - 1,
                    cuts[c],
                    kind,
                    n
                );
            }
        }
        let interior_only = kept
            .iter()
            .filter(|h| cuts[h.cut_lo] > 0 && cuts[h.cut_hi] < pg.len())
            .count();
        println!(
            "  equations using ONLY interior cuts (real mixing leak): {}",
            interior_only
        );
        if degree >= 2 {
            let withprod = kept.iter().filter(|h| h.n_prod_terms > 0).count();
            println!(
                "  uses >=1 product    : {} ({:.1}%)",
                withprod,
                100.0 * withprod as f64 / kept.len() as f64
            );
        }
    }
    eprintln!("done ({:.1}s)", t0.elapsed().as_secs_f64());
}
