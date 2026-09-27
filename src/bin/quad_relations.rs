//! DEGREE-2 analogue of invariant_classify: count affine AND quadratic relations
//! between two circuit states, to see whether quadratic memory outlives linear
//! memory.
//!
//! The full degree-2 space over 512x2 coordinates has C(1024,2)+1025 ~ 524k
//! monomials — it needs >524k samples and an infeasible elimination. So we fix a
//! random window of W wires, observed at BOTH cuts (2W coordinates), and build
//! the exact monomial set
//!     1  |  v_1..v_2W  |  v_i*v_j (i<j)
//! whose size 1 + 2W + C(2W,2) stays small enough for exact GF(2) elimination.
//! Relations = monomials - rank, all re-verified on fresh samples.
//!
//! Reported separately:
//!   deg1 = relations using only the constant + linear monomials
//!   deg2 = relations over the whole monomial set
//! Note deg2 is inflated by products of deg1 relations (L=0 implies L*v=0), so
//! the decisive readout is a gap where deg1 = 0 but deg2 > 0: there, every
//! surviving relation is genuinely quadratic.
//!
//!   quad_relations <circuit.mpmct1> [--wires W] [--samples S] [--verify V]
//!                  [--cut-lo F] [--cut-hi F] [--wire-seed N]
use local_mixing::postmix::format::read_mpmct;
use local_mixing::postmix::xgate::XGate;
use std::time::Instant;

fn snapshot(
    gates: &[XGate],
    num_wires: usize,
    xs: &[u128],
    blk: usize,
    cuts: &[usize],
) -> Vec<Vec<u64>> {
    let mut state = vec![0u64; num_wires];
    for (lane, &x) in xs.iter().enumerate() {
        for i in 0..blk {
            if (x >> i) & 1 == 1 {
                state[i] |= 1u64 << lane;
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
    let path = args.next().expect("usage: quad_relations <circuit.mpmct1> [opts]");
    let (mut w_sub, mut samples, mut verify_n) = (32usize, 8192usize, 2048usize);
    let (mut cut_lo, mut cut_hi) = (0.05f64, 0.10f64);
    let (mut wire_seed, mut blk) = (12345u64, 128usize);
    let mut it = args.peekable();
    while let Some(a) = it.next() {
        match a.as_str() {
            "--wires" => w_sub = it.next().unwrap().parse().unwrap(),
            "--samples" => samples = it.next().unwrap().parse().unwrap(),
            "--verify" => verify_n = it.next().unwrap().parse().unwrap(),
            "--cut-lo" => cut_lo = it.next().unwrap().parse().unwrap(),
            "--cut-hi" => cut_hi = it.next().unwrap().parse().unwrap(),
            "--wire-seed" => wire_seed = it.next().unwrap().parse().unwrap(),
            "--blk" => blk = it.next().unwrap().parse().unwrap(),
            o => panic!("unknown arg {o}"),
        }
    }
    assert!(samples % 64 == 0 && verify_n % 64 == 0);

    let t0 = Instant::now();
    let (gates, num_wires) = read_mpmct(&path).expect("read mpmct");
    let ng = gates.len();
    let cuts: Vec<usize> =
        vec![(ng as f64 * cut_lo).round() as usize, (ng as f64 * cut_hi).round() as usize];

    // deterministic wire subset (same wires across every circuit compared)
    let mut ws = wire_seed;
    let mut nxt = move || -> u64 {
        ws = ws.wrapping_add(0x9E3779B97F4A7C15);
        let mut z = ws;
        z = (z ^ (z >> 30)).wrapping_mul(0xBF58476D1CE4E5B9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94D049BB133111EB);
        z ^ (z >> 31)
    };
    let mut chosen: Vec<usize> = Vec::new();
    while chosen.len() < w_sub {
        let c = (nxt() % num_wires as u64) as usize;
        if !chosen.contains(&c) {
            chosen.push(c);
        }
    }
    chosen.sort_unstable();

    let ncoord = 2 * w_sub;
    let nprod = ncoord * (ncoord - 1) / 2;
    let nmono = 1 + ncoord + nprod;
    let sw = samples / 64;
    eprintln!(
        "{ng} gates; cuts {:?}; W={w_sub} wires -> {ncoord} coords, {nmono} monomials, {samples} samples",
        cuts
    );
    assert!(samples > nmono, "need samples > monomials ({nmono})");

    // sample RNG (splitmix64: NOT GF(2)-linear)
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
    let blk_mask: u128 = if blk >= 128 { u128::MAX } else { (1u128 << blk) - 1 };

    // ---- coordinate signatures ----
    let mut coord = vec![0u64; ncoord * sw];
    for wi in 0..sw {
        let xs: Vec<u128> = (0..64).map(|_| rx() & blk_mask).collect();
        let snaps = snapshot(&gates, num_wires, &xs, blk, &cuts);
        for (ci, snap) in snaps.iter().enumerate() {
            for (j, &w) in chosen.iter().enumerate() {
                coord[(ci * w_sub + j) * sw + wi] = snap[w];
            }
        }
    }

    // monomial signature by index: 0 = const, 1..ncoord = linear, then products
    let prod_pairs: Vec<(usize, usize)> = {
        let mut v = Vec::with_capacity(nprod);
        for i in 0..ncoord {
            for j in (i + 1)..ncoord {
                v.push((i, j));
            }
        }
        v
    };
    let mono = |idx: usize, out: &mut Vec<u64>| {
        out.clear();
        if idx == 0 {
            out.extend(std::iter::repeat(!0u64).take(sw));
        } else if idx <= ncoord {
            let c = idx - 1;
            out.extend_from_slice(&coord[c * sw..(c + 1) * sw]);
        } else {
            let (i, j) = prod_pairs[idx - ncoord - 1];
            for k in 0..sw {
                out.push(coord[i * sw + k] & coord[j * sw + k]);
            }
        }
    };

    // ---- eliminate in index order, recording the deg-1 count partway ----
    let tw = nmono.div_ceil(64);
    let mut piv: Vec<i32> = vec![-1; samples];
    let mut bas_sig: Vec<Vec<u64>> = Vec::new();
    let mut bas_tag: Vec<Vec<u64>> = Vec::new();
    let mut rels: Vec<Vec<u64>> = Vec::new();
    let mut deg1_rels = 0usize;
    let mut buf: Vec<u64> = Vec::with_capacity(sw);

    for idx in 0..nmono {
        if idx == ncoord + 1 {
            deg1_rels = rels.len(); // everything so far used only const+linear
        }
        mono(idx, &mut buf);
        let mut v = buf.clone();
        let mut tag = vec![0u64; tw];
        tag[idx / 64] |= 1u64 << (idx % 64);
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
    if nmono <= ncoord + 1 {
        deg1_rels = rels.len();
    }
    let rank = bas_sig.len();

    // ---- verify every relation on FRESH samples ----
    let supports: Vec<Vec<usize>> = rels
        .iter()
        .map(|t| (0..nmono).filter(|&m| (t[m / 64] >> (m % 64)) & 1 == 1).collect())
        .collect();
    let mut bad = 0usize;
    for _ in 0..(verify_n / 64) {
        let xs: Vec<u128> = (0..64).map(|_| rx() & blk_mask).collect();
        let snaps = snapshot(&gates, num_wires, &xs, blk, &cuts);
        let mut cv = vec![0u64; ncoord];
        for (ci, snap) in snaps.iter().enumerate() {
            for (j, &w) in chosen.iter().enumerate() {
                cv[ci * w_sub + j] = snap[w];
            }
        }
        let val = |m: usize| -> u64 {
            if m == 0 {
                !0u64
            } else if m <= ncoord {
                cv[m - 1]
            } else {
                let (i, j) = prod_pairs[m - ncoord - 1];
                cv[i] & cv[j]
            }
        };
        for sup in &supports {
            let mut acc = 0u64;
            for &m in sup {
                acc ^= val(m);
            }
            if acc != 0 {
                bad += 1;
                break;
            }
        }
    }

    println!(
        "W={} coords={} monomials={} rank={} deg1_relations={} deg2_relations={} verify_fail_batches={} ({:.1}s)",
        w_sub,
        ncoord,
        nmono,
        rank,
        deg1_rels,
        rels.len(),
        bad,
        t0.elapsed().as_secs_f64()
    );
}
