//! Extract, verify and CLASSIFY the cross-cut affine invariants of a mixed
//! circuit — the structure a SAT-invariant attack would inject.
//!
//! A relation is a set S of coordinates (cut, wire) plus a constant bit c with
//!     XOR_{(k,w) in S} wire_w(cut k)  =  c      for every zero-slice input x.
//!
//! Found as the GF(2) nullspace of the coordinate signature matrix (signature =
//! the coordinate's value across S random inputs), tracked with tag vectors so
//! we recover the relations themselves, not just their count. Then:
//!   * VERIFIED on fresh independent samples (guards against sampling artifacts)
//!   * classified: trivial pinned-input constants vs genuine cross-wire structure
//!   * LEAK CHECK: any relation supported only on input x-bits and output payload
//!     bits is a direct linear y<->x break.
//!
//!   invariant_classify <circuit.mpmct1> [--cuts K] [--samples S] [--verify V]
use local_mixing::postmix::format::read_mpmct;
use local_mixing::postmix::xgate::XGate;
use std::time::Instant;

fn snapshot(
    gates: &[XGate],
    num_wires: usize,
    xs: &[u128],
    xw0: usize,
    blk: usize,
    cuts: &[usize],
) -> Vec<Vec<u64>> {
    let mut state = vec![0u64; num_wires];
    for (lane, &x) in xs.iter().enumerate() {
        for i in 0..blk {
            if (x >> i) & 1 == 1 {
                state[xw0 + i] |= 1u64 << lane;
            }
        }
    }
    let mut snaps = Vec::with_capacity(cuts.len());
    let mut ci = 0usize;
    for (gi, g) in gates.iter().enumerate() {
        while ci < cuts.len() && cuts[ci] == gi {
            snaps.push(state.clone());
            ci += 1;
        }
        // every cut captured: no need to run the remaining gates
        if ci == cuts.len() {
            break;
        }
        g.apply_lanes(&mut state);
    }
    while ci < cuts.len() {
        snaps.push(state.clone());
        ci += 1;
    }
    snaps
}

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

fn main() {
    let mut args = std::env::args().skip(1);
    let path = args.next().expect("usage: invariant_classify <circuit.mpmct1> [opts]");
    let (mut ncuts, mut samples, mut verify_n) = (8usize, 8192usize, 4096usize);
    let (mut xw0, mut blk, mut yw0) = (0usize, 128usize, 128usize);
    let mut dump: Option<String> = None;
    let (mut cut_lo, mut cut_hi) = (0.0f64, 1.0f64);
    let mut include_input = false;
    let mut it = args.peekable();
    while let Some(a) = it.next() {
        match a.as_str() {
            "--cuts" => ncuts = it.next().unwrap().parse().unwrap(),
            "--samples" => samples = it.next().unwrap().parse().unwrap(),
            "--verify" => verify_n = it.next().unwrap().parse().unwrap(),
            "--xw0" => xw0 = it.next().unwrap().parse().unwrap(),
            "--yw0" => yw0 = it.next().unwrap().parse().unwrap(),
            "--blk" => blk = it.next().unwrap().parse().unwrap(),
            "--dump" => dump = Some(it.next().unwrap()),
            // restrict cuts to a depth window [lo,hi] as fractions of the circuit,
            // so cuts can be concentrated where structure is suspected (e.g. the
            // tail, where the payload block is finalised)
            "--cut-lo" => cut_lo = it.next().unwrap().parse().unwrap(),
            "--cut-hi" => cut_hi = it.next().unwrap().parse().unwrap(),
            // always include gate 0 (the TRUE input) as cut 0, so the x<->y leak
            // check stays meaningful when the window is pushed to the tail
            "--include-input" => include_input = true,
            o => panic!("unknown arg {o}"),
        }
    }
    assert!(samples % 64 == 0 && verify_n % 64 == 0);

    let t0 = Instant::now();
    let (gates, num_wires) = read_mpmct(&path).expect("read mpmct");
    let ng = gates.len();
    let (g_lo, g_hi) = (ng as f64 * cut_lo, ng as f64 * cut_hi);
    let mut cuts: Vec<usize> = (0..=ncuts)
        .map(|i| (g_lo + (g_hi - g_lo) * i as f64 / ncuts as f64).round() as usize)
        .collect();
    if include_input && cuts[0] != 0 {
        cuts.insert(0, 0);
    }
    let ncut = cuts.len();
    let dim = ncut * num_wires; // coordinate D itself = the constant (all-ones)
    let sw = samples / 64; // signature words
    let tw = (dim + 1).div_ceil(64); // tag words
    eprintln!(
        "{} gates, {} wires; {} cuts x {} wires = {} coords; {} samples ({:.1}s)",
        ng, num_wires, ncut, num_wires, dim, samples, t0.elapsed().as_secs_f64()
    );
    assert!(samples > dim, "need samples > coords ({dim})");

    // NOTE: the sampler must NOT be GF(2)-linear. An xorshift generator is linear
    // over GF(2), so its whole output stream lies in a linearly-structured set and
    // affine relations can hold across every sample without holding globally —
    // manufacturing false "invariants" (caught by the X* guard in preimage_cnf).
    // splitmix64 is nonlinear (odd-multiplier mixing).
    let mut rng: u64 = 0x243F6A8885A308D3;
    let mut next_rand = move || -> u128 {
        let mut sm = || -> u64 {
            rng = rng.wrapping_add(0x9E3779B97F4A7C15);
            let mut z = rng;
            z = (z ^ (z >> 30)).wrapping_mul(0xBF58476D1CE4E5B9);
            z = (z ^ (z >> 27)).wrapping_mul(0x94D049BB133111EB);
            z ^ (z >> 31)
        };
        let hi = sm();
        let lo = sm();
        ((hi as u128) << 64) | lo as u128
    };
    let blk_mask: u128 = if blk >= 128 { u128::MAX } else { (1u128 << blk) - 1 };

    // ---- signatures ----
    let mut sig = vec![0u64; (dim + 1) * sw];
    for w in 0..sw {
        sig[dim * sw + w] = !0u64; // constant coordinate
    }
    for wi in 0..sw {
        let xs: Vec<u128> = (0..64).map(|_| next_rand() & blk_mask).collect();
        let snaps = snapshot(&gates, num_wires, &xs, xw0, blk, &cuts);
        for (ci, snap) in snaps.iter().enumerate() {
            for w in 0..num_wires {
                sig[(ci * num_wires + w) * sw + wi] = snap[w];
            }
        }
    }
    eprintln!("signatures collected ({:.1}s)", t0.elapsed().as_secs_f64());

    // ---- eliminate, tracking tags -> nullspace = relations ----
    let mut piv: Vec<i32> = vec![-1; samples];
    let mut bas_sig: Vec<Vec<u64>> = Vec::new();
    let mut bas_tag: Vec<Vec<u64>> = Vec::new();
    let mut rels: Vec<Vec<u64>> = Vec::new();

    for coord in 0..=dim {
        let mut v: Vec<u64> = sig[coord * sw..(coord + 1) * sw].to_vec();
        let mut tag = vec![0u64; tw];
        tag[coord / 64] |= 1u64 << (coord % 64);
        loop {
            match lead(&v) {
                None => {
                    rels.push(tag);
                    break;
                }
                Some(p) => {
                    if piv[p] >= 0 {
                        let i = piv[p] as usize;
                        xor_into(&mut v, &bas_sig[i]);
                        xor_into(&mut tag, &bas_tag[i]);
                    } else {
                        piv[p] = bas_sig.len() as i32;
                        bas_sig.push(v);
                        bas_tag.push(tag);
                        break;
                    }
                }
            }
        }
    }
    let rank = bas_sig.len();
    eprintln!("rank {} -> {} relations ({:.1}s)", rank, rels.len(), t0.elapsed().as_secs_f64());

    // ---- RREF the relation space for sparse, canonical relations ----
    let mut rpiv: Vec<i32> = vec![-1; dim + 1];
    let mut rows: Vec<Vec<u64>> = Vec::new();
    for r in rels {
        let mut v = r;
        loop {
            match lead(&v) {
                None => break,
                Some(p) => {
                    if rpiv[p] >= 0 {
                        let i = rpiv[p] as usize;
                        let src = rows[i].clone();
                        xor_into(&mut v, &src);
                    } else {
                        rpiv[p] = rows.len() as i32;
                        rows.push(v);
                        break;
                    }
                }
            }
        }
    }
    // back-reduce so each pivot appears in exactly one row
    for i in 0..rows.len() {
        for j in 0..rows.len() {
            if i == j {
                continue;
            }
            if let Some(p) = lead(&rows[i]) {
                if (rows[j][p / 64] >> (p % 64)) & 1 == 1 {
                    let src = rows[i].clone();
                    xor_into(&mut rows[j], &src);
                }
            }
        }
    }

    // ---- verify on FRESH samples ----
    let supports: Vec<Vec<usize>> = rows
        .iter()
        .map(|t| {
            (0..dim).filter(|&c| (t[c / 64] >> (c % 64)) & 1 == 1).collect::<Vec<_>>()
        })
        .collect();
    let consts: Vec<bool> = rows.iter().map(|t| (t[dim / 64] >> (dim % 64)) & 1 == 1).collect();

    let mut failed = vec![false; rows.len()];
    for _ in 0..(verify_n / 64) {
        let xs: Vec<u128> = (0..64).map(|_| next_rand() & blk_mask).collect();
        let snaps = snapshot(&gates, num_wires, &xs, xw0, blk, &cuts);
        for (ri, sup) in supports.iter().enumerate() {
            let mut acc = 0u64;
            for &c in sup {
                acc ^= snaps[c / num_wires][c % num_wires];
            }
            let want = if consts[ri] { !0u64 } else { 0u64 };
            if acc != want {
                failed[ri] = true;
            }
        }
    }
    let nfail = failed.iter().filter(|&&f| f).count();
    let good: Vec<usize> = (0..rows.len()).filter(|&i| !failed[i]).collect();
    eprintln!("verified on {} fresh samples ({:.1}s)\n", verify_n, t0.elapsed().as_secs_f64());

    // ---- classify ----
    let last_cut = ncut - 1;
    let is_x = |c: usize| c / num_wires == 0 && c % num_wires >= xw0 && c % num_wires < xw0 + blk;
    let is_pinned_in = |c: usize| c / num_wires == 0 && !is_x(c);
    let is_payload_out =
        |c: usize| c / num_wires == last_cut && c % num_wires >= yw0 && c % num_wires < yw0 + blk;

    let mut trivial = 0usize; // only pinned zero input wires
    let mut touch_x = 0usize;
    let mut touch_payload = 0usize;
    let mut leak = 0usize; // supported ONLY on x-in and payload-out => linear break
    let mut interior = 0usize;
    let mut sizes: Vec<usize> = Vec::new();
    let mut cutspan: Vec<usize> = Vec::new();

    for &i in &good {
        let sup = &supports[i];
        sizes.push(sup.len());
        let cs: std::collections::BTreeSet<usize> = sup.iter().map(|&c| c / num_wires).collect();
        cutspan.push(cs.len());
        if sup.iter().all(|&c| is_pinned_in(c)) {
            trivial += 1;
            continue;
        }
        let tx = sup.iter().any(|&c| is_x(c));
        let tp = sup.iter().any(|&c| is_payload_out(c));
        if tx {
            touch_x += 1;
        }
        if tp {
            touch_payload += 1;
        }
        if sup.iter().all(|&c| is_x(c) || is_payload_out(c)) && tp {
            leak += 1;
        }
        interior += 1;
    }

    println!("=== cross-cut affine invariant classification ===");
    println!("circuit            : {}", path);
    println!("gates / wires      : {} / {}", ng, num_wires);
    println!("cuts               : {} at {:?}", ncut, cuts);
    println!("coords / samples   : {} / {}", dim, samples);
    println!("signature rank     : {}", rank);
    println!("relations (raw)    : {}", rows.len());
    println!("failed fresh verify: {}  <- sampling artifacts, discarded", nfail);
    println!("relations (VERIFIED): {}", good.len());
    println!();
    println!("  trivial (pinned zero input wires only) : {}", trivial);
    println!("  NON-TRIVIAL structural relations       : {}", interior);
    println!("    ...touching input x bits             : {}", touch_x);
    println!("    ...touching output payload bits      : {}", touch_payload);
    println!("  *** LINEAR y<->x LEAK relations ***    : {}", leak);
    if !sizes.is_empty() {
        let mut s = sizes.clone();
        s.sort_unstable();
        println!();
        println!(
            "  support size: min {}, median {}, max {}",
            s[0],
            s[s.len() / 2],
            s[s.len() - 1]
        );
        let mut cspan = cutspan.clone();
        cspan.sort_unstable();
        println!(
            "  cuts spanned: min {}, median {}, max {}",
            cspan[0],
            cspan[cspan.len() / 2],
            cspan[cspan.len() - 1]
        );
    }

    // show the smallest non-trivial relations
    let mut nt: Vec<usize> =
        good.iter().copied().filter(|&i| !supports[i].iter().all(|&c| is_pinned_in(c))).collect();
    nt.sort_by_key(|&i| supports[i].len());
    println!("\n  smallest non-trivial relations (cut:wire):");
    for &i in nt.iter().take(12) {
        let s: Vec<String> = supports[i]
            .iter()
            .take(8)
            .map(|&c| format!("{}:{}", c / num_wires, c % num_wires))
            .collect();
        println!(
            "    [{} terms] {}{} = {}",
            supports[i].len(),
            s.join(" ^ "),
            if supports[i].len() > 8 { " ^ ..." } else { "" },
            consts[i] as u8
        );
    }
    // dump VERIFIED relations for injection into the preimage CNF
    if let Some(p) = dump {
        use std::io::Write;
        let mut out = String::new();
        out.push_str("# cuts");
        for c in &cuts {
            out.push_str(&format!(" {c}"));
        }
        out.push('\n');
        let mut ndump = 0usize;
        for &i in &good {
            if supports[i].iter().all(|&c| is_pinned_in(c)) {
                continue; // solver gets these from unit propagation anyway
            }
            out.push_str(&format!("rel {} {}", consts[i] as u8, supports[i].len()));
            for &c in &supports[i] {
                out.push_str(&format!(" {}:{}", c / num_wires, c % num_wires));
            }
            out.push('\n');
            ndump += 1;
        }
        std::fs::File::create(&p).unwrap().write_all(out.as_bytes()).unwrap();
        println!("\n  dumped {ndump} non-trivial verified relations -> {p}");
    }

    if leak > 0 {
        println!("\n  !!! {} relation(s) link input x bits directly to output payload bits.", leak);
        println!("      That is a LINEAR inversion constraint — feed these to the solver.");
    } else {
        println!("\n  No relation links input x bits to output payload bits:");
        println!("  the invariants are interior-only and give NO linear handle on the preimage.");
    }
}
