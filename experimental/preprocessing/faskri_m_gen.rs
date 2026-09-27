//! FASKRI + M-lane prototype — Fixed Ambient Skeleton, Keyed Routing, Fused
//! Injection, WITH a withheld nonlinear mask lane.
//!
//! Extends `faskri_gen.rs` (docs/FASKRI_PROTOTYPE_20260826.md), whose §5/§7 say the
//! load-bearing open question is the M-lane: Phi alone cannot hide anything because
//! a groupwise frame's domain is only `2^g` and the trace adversary gets unlimited
//! views of it (docs/FASKRI_PROTOTYPE_20260826.md §6a). What lifts the effective
//! domain beyond `2^g` is an x-dependent term that is NEVER materialized.
//!
//! This binary carries the source state as `z_G = S_G(s|coords(G)) ⊕ m_G(band)`,
//! where:
//!   * `band` is a small pool of extra wires filled ONCE from x by keyed linear
//!     CNOTs and never rewritten (so it stays x-dependent even on the honest zero
//!     slice — the failure mode of FASKRI §6c is closed by construction);
//!   * `m_G` is a keyed g-bit mask that is a NONLINEAR (algebraic degree ≥ 2)
//!     function of the band wires. The mask is a product of band wires; the product
//!     value is never written to any wire — it is computed inline inside each block
//!     from the band wires read as controls.
//!
//! Every block (head / inject / surgery / refresh / tail) is re-synthesized
//! monolithically over its `(core + band)` domain as: unmask → core transform →
//! remask. The unmasked value only ever exists transiently INSIDE a block. Because
//! the mask cancels head-to-tail and every body block carries `⊕m` on both sides,
//! the emitted circuit is functionally EQUAL to the source sandwich on the payload
//! wires, so `A(x,0) = (junk, C(x))` is preserved exactly — checked by `--verify`
//! comparing ONLY the payload wires.
//!
//! Arms (`--m-lane`):
//!   * `idle` — band wires present and filled, mask ≡ 0. Isolates the effect of
//!     widening alone; should reproduce faskri_gen's leak.
//!   * `prod` — keyed nonlinear product mask. The candidate.
//!
//! The decisive A/B is: same seed, same `--gates` prefix, `idle` vs `prod`, scored
//! by `segment_deduce --pred-cuts 64 --degree 1` (interior-only deducible count).
//!
//! Usage:
//!   faskri_m_gen --source <sandwich.mpmct1> --out <p.mpmct1>
//!                [--g 3] [--nband 4] [--m-lane idle|prod] [--gates N]
//!                [--dummy K] [--seed S] [--verify N]

use local_mixing::circuit::xgate::XGate;
use local_mixing::engine::format::{read_mpmct, write_mpmct};
use std::collections::HashMap;
use std::time::Instant;

// ---------------------------------------------------------------- rng

struct Rng(u64);
impl Rng {
    fn next(&mut self) -> u64 {
        self.0 = self.0.wrapping_add(0x9E3779B97F4A7C15);
        let mut z = self.0;
        z = (z ^ (z >> 30)).wrapping_mul(0xBF58476D1CE4E5B9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94D049BB133111EB);
        z ^ (z >> 31)
    }
    fn below(&mut self, n: usize) -> usize {
        (self.next() % n as u64) as usize
    }
}

// ---------------------------------------------------------------- small perms

type Perm = Vec<u16>;

fn identity(w: usize) -> Perm {
    (0..(1u32 << w) as u16).collect()
}

fn invert(p: &Perm) -> Perm {
    let mut q = vec![0u16; p.len()];
    for (i, &v) in p.iter().enumerate() {
        q[v as usize] = i as u16;
    }
    q
}

/// Algebraic degree of each output coordinate, via the Moebius transform.
fn coord_degrees(p: &Perm, w: usize) -> Vec<usize> {
    let n = 1usize << w;
    let mut deg = vec![0usize; w];
    for bit in 0..w {
        let mut f: Vec<u8> = (0..n).map(|i| ((p[i] >> bit) & 1) as u8).collect();
        let mut step = 1;
        while step < n {
            for base in (0..n).step_by(step << 1) {
                for i in base..base + step {
                    f[i + step] ^= f[i];
                }
            }
            step <<= 1;
        }
        deg[bit] = (0..n)
            .filter(|&i| f[i] == 1)
            .map(|i| (i as u32).count_ones() as usize)
            .max()
            .unwrap_or(0);
    }
    deg
}

/// Keyed `w`-bit S-box whose INVERSE has algebraic degree exactly `w-1` in EVERY
/// output coordinate (ENCODED_CHAIN_DESIGN §7.1).
fn draw_sbox(rng: &mut Rng, w: usize) -> Perm {
    let n = 1usize << w;
    loop {
        let mut p: Perm = identity(w);
        for i in (1..n).rev() {
            let j = rng.below(i + 1);
            p.swap(i, j);
        }
        if coord_degrees(&invert(&p), w).iter().all(|&d| d == w - 1) {
            return p;
        }
    }
}

/// Algebraic degree of output bit `bit` of a band->g mask table over `nband` bits.
fn mask_bit_degree(tab: &[u16], nband: usize, bit: usize) -> usize {
    let n = 1usize << nband;
    let mut f: Vec<u8> = (0..n).map(|i| ((tab[i] >> bit) & 1) as u8).collect();
    let mut step = 1;
    while step < n {
        for base in (0..n).step_by(step << 1) {
            for i in base..base + step {
                f[i + step] ^= f[i];
            }
        }
        step <<= 1;
    }
    (0..n)
        .filter(|&i| f[i] == 1)
        .map(|i| (i as u32).count_ones() as usize)
        .max()
        .unwrap_or(0)
}

/// Keyed mask table: `band in 0..2^nband  ->  g-bit mask`, with EVERY output bit
/// algebraic degree ≥ 2 in the band wires (so the mask is genuinely nonlinear and
/// cannot be an affine — i.e. additively transparent — function of the band, the
/// negative result of DRAIN_SET.md).
fn draw_mask(rng: &mut Rng, gw: usize, nband: usize) -> Vec<u16> {
    let n = 1usize << nband;
    let m = ((1u32 << gw) - 1) as u16;
    loop {
        let tab: Vec<u16> = (0..n).map(|_| (rng.next() as u16) & m).collect();
        if (0..gw).all(|b| mask_bit_degree(&tab, nband, b) >= 2) {
            return tab;
        }
    }
}

// ---------------------------------------------------------------- MMD synthesis

#[derive(Clone, Debug)]
struct McGate {
    target: u8,
    ctrls: Vec<u8>,
}

fn apply_mc(f: &mut [u16], g: &McGate) {
    let mask: u16 = g.ctrls.iter().fold(0u16, |m, &c| m | (1 << c));
    let tb: u16 = 1 << g.target;
    for v in f.iter_mut() {
        if (*v & mask) == mask {
            *v ^= tb;
        }
    }
}

/// Miller-Maslov-Dueck transformation-based synthesis into multi-control gates.
fn mmd(perm: &Perm, w: usize) -> Vec<McGate> {
    let n = 1usize << w;
    let mut f = perm.clone();
    let mut gates: Vec<McGate> = Vec::new();
    for i in 0..n {
        let v = f[i] as usize;
        if v == i {
            continue;
        }
        let mut cur = v;
        for b in 0..w {
            if (i >> b) & 1 == 1 && (cur >> b) & 1 == 0 {
                let ctrls: Vec<u8> =
                    (0..w).filter(|&c| (cur >> c) & 1 == 1).map(|c| c as u8).collect();
                let g = McGate { target: b as u8, ctrls };
                apply_mc(&mut f, &g);
                gates.push(g);
                cur |= 1 << b;
            }
        }
        for b in 0..w {
            if (cur >> b) & 1 == 1 && (i >> b) & 1 == 0 {
                let ctrls: Vec<u8> =
                    (0..w).filter(|&c| (i >> c) & 1 == 1).map(|c| c as u8).collect();
                let g = McGate { target: b as u8, ctrls };
                apply_mc(&mut f, &g);
                gates.push(g);
                cur &= !(1 << b);
            }
        }
        debug_assert_eq!(f[i] as usize, i);
    }
    gates.reverse();
    gates
}

fn simulate_mc(gates: &[McGate], w: usize) -> Perm {
    let n = 1usize << w;
    (0..n)
        .map(|x| {
            let mut v = x as u16;
            for g in gates {
                let mask: u16 = g.ctrls.iter().fold(0u16, |m, &c| m | (1 << c));
                if (v & mask) == mask {
                    v ^= 1 << g.target;
                }
            }
            v
        })
        .collect()
}

/// Shorter of the forward and inverse syntheses; always verified.
fn synth(perm: &Perm, w: usize) -> Vec<McGate> {
    let fwd = mmd(perm, w);
    let mut bwd = mmd(&invert(perm), w);
    bwd.reverse();
    let best = if bwd.len() < fwd.len() { bwd } else { fwd };
    assert_eq!(&simulate_mc(&best, w), perm, "synthesis verification failed");
    best
}

// ---------------------------------------------------------------- ext-domain wrap

/// Extend a core permutation `t` (width `w`) to the `(w + nband)`-bit domain that
/// also reads the band wires. The band bits pass through unchanged (so MMD never
/// targets them); the core is optionally unmasked on the way in and remasked on
/// the way out by `m[band]`.
fn ext_single(t: &Perm, w: usize, nband: usize, m: &[u16], unmask_in: bool, remask_out: bool) -> Perm {
    let cn = 1usize << w;
    let n = 1usize << (w + nband);
    (0..n)
        .map(|idx| {
            let core = idx & (cn - 1);
            let band = idx >> w;
            let mm = m[band] as usize;
            let cin = if unmask_in { core ^ mm } else { core };
            let cout = t[cin] as usize;
            let cout = if remask_out { cout ^ mm } else { cout };
            (cout | (band << w)) as u16
        })
        .collect()
}

/// Two-group variant (surgery): core is `(wa + wb)` bits split at `wa`, each half
/// unmasked/remasked by its own group mask.
fn ext_pair(
    t: &Perm,
    wa: usize,
    wb: usize,
    nband: usize,
    ma: &[u16],
    mb: &[u16],
) -> Perm {
    let wc = wa + wb;
    let cn = 1usize << wc;
    let camask = (1usize << wa) - 1;
    let n = 1usize << (wc + nband);
    (0..n)
        .map(|idx| {
            let core = idx & (cn - 1);
            let band = idx >> wc;
            let (mav, mbv) = (ma[band] as usize, mb[band] as usize);
            let ca = core & camask;
            let cb = core >> wa;
            let u = (ca ^ mav) | ((cb ^ mbv) << wa);
            let v = t[u] as usize;
            let va = v & camask;
            let vb = v >> wa;
            let out = (va ^ mav) | ((vb ^ mbv) << wa);
            (out | (band << wc)) as u16
        })
        .collect()
}

// ---------------------------------------------------------------- source gates

#[derive(Clone)]
struct SrcGate {
    target: usize,
    comp: bool,
    ctrls: Vec<(usize, bool)>,
}

impl SrcGate {
    fn support(&self) -> Vec<usize> {
        let mut v = vec![self.target];
        for &(w, _) in &self.ctrls {
            if !v.contains(&w) {
                v.push(w);
            }
        }
        v
    }
}

// ---------------------------------------------------------------- fabric

struct Fabric {
    n: usize,
    nband: usize,
    band_wires: Vec<usize>,
    band_src: Vec<Vec<usize>>,
    groups: Vec<Vec<usize>>,
    wire_group: Vec<usize>,
    slot_coord: Vec<usize>,
    coord_slot: Vec<usize>,
    sbox: Vec<Perm>,
    mask: Vec<Vec<u16>>,
    gates: Vec<XGate>,
    n_inject: usize,
    n_surgery: usize,
    n_refresh: usize,
}

impl Fabric {
    /// Partition `n` payload wires into groups of `g` (remainder folded into the
    /// last); allocate `nband` band wires at `n..n+nband`; draw per-group masks
    /// (`prod`) or zero masks (`idle`).
    fn new(n: usize, g: usize, nband: usize, prod: bool, blk: usize, rng: &mut Rng) -> Self {
        let mut groups: Vec<Vec<usize>> = Vec::new();
        let full = n / g;
        for k in 0..full {
            groups.push((k * g..(k + 1) * g).collect());
        }
        if n % g != 0 {
            let last = groups.last_mut().expect("at least one group");
            last.extend(full * g..n);
        }
        let mut wire_group = vec![0usize; n];
        for (gi, ws) in groups.iter().enumerate() {
            for &w in ws {
                wire_group[w] = gi;
            }
        }
        let sbox = groups.iter().map(|ws| identity(ws.len())).collect();
        let band_wires: Vec<usize> = (n..n + nband).collect();
        // Each band wire = XOR of `fanin` distinct x-wires (0..blk), keyed. Linear
        // in x, so the band itself is affine — the nonlinearity lives entirely in
        // the mask, whose product terms are what the trace adversary cannot form.
        let fanin = 3.min(blk);
        let band_src: Vec<Vec<usize>> = (0..nband)
            .map(|_| {
                let mut s: Vec<usize> = Vec::new();
                while s.len() < fanin {
                    let x = rng.below(blk);
                    if !s.contains(&x) {
                        s.push(x);
                    }
                }
                s
            })
            .collect();
        let mask: Vec<Vec<u16>> = groups
            .iter()
            .map(|ws| {
                if prod {
                    draw_mask(rng, ws.len(), nband)
                } else {
                    vec![0u16; 1usize << nband]
                }
            })
            .collect();
        Fabric {
            n,
            nband,
            band_wires,
            band_src,
            groups,
            wire_group,
            slot_coord: (0..n).collect(),
            coord_slot: (0..n).collect(),
            sbox,
            mask,
            gates: Vec::new(),
            n_inject: 0,
            n_surgery: 0,
            n_refresh: 0,
        }
    }

    #[inline]
    fn gw(&self, gi: usize) -> usize {
        self.groups[gi].len()
    }
    fn pos_in_group(&self, wire: usize) -> usize {
        let gi = self.wire_group[wire];
        self.groups[gi].iter().position(|&w| w == wire).expect("wire in its group")
    }

    fn push(&mut self, wires: &[usize], mc: &[McGate]) {
        for g in mc {
            let lits: Vec<(u16, bool)> =
                g.ctrls.iter().map(|&c| (wires[c as usize] as u16, true)).collect();
            let target = wires[g.target as usize] as u16;
            if let Some(x) = XGate::conj(target, lits) {
                self.gates.push(x);
            }
        }
    }

    /// Emit the one-time band fill: `band[k] ^= x-wire` for each keyed source.
    fn band_fill(&mut self) {
        for (k, srcs) in self.band_src.clone().iter().enumerate() {
            let bw = self.band_wires[k] as u16;
            for &x in srcs {
                self.gates.push(XGate::cnot(bw, x as u16));
            }
        }
    }

    fn single_wires(&self, gi: usize) -> Vec<usize> {
        let mut wires = self.groups[gi].clone();
        wires.extend(self.band_wires.iter().copied());
        wires
    }

    fn head_encode(&mut self, rng: &mut Rng) {
        for gi in 0..self.groups.len() {
            let w = self.gw(gi);
            let s = draw_sbox(rng, w);
            // input is the RAW coord, output is masked: no unmask, remask.
            let ext = ext_single(&s, w, self.nband, &self.mask[gi], false, true);
            let mc = synth(&ext, w + self.nband);
            let wires = self.single_wires(gi);
            self.push(&wires, &mc);
            self.sbox[gi] = s;
        }
    }

    fn refresh(&mut self, gi: usize, rng: &mut Rng) {
        let w = self.gw(gi);
        let s_inv = invert(&self.sbox[gi]);
        let s_new = draw_sbox(rng, w);
        let t: Perm = (0..(1usize << w)).map(|z| s_new[s_inv[z] as usize]).collect();
        let ext = ext_single(&t, w, self.nband, &self.mask[gi], true, true);
        let mc = synth(&ext, w + self.nband);
        let wires = self.single_wires(gi);
        self.push(&wires, &mc);
        self.sbox[gi] = s_new;
        self.n_refresh += 1;
    }

    fn surgery(&mut self, pa: usize, pb: usize, rng: &mut Rng) {
        let (ga, gb) = (self.wire_group[pa], self.wire_group[pb]);
        assert_ne!(ga, gb);
        let (ia, ib) = (self.pos_in_group(pa), self.pos_in_group(pb));
        let (wa, wb) = (self.gw(ga), self.gw(gb));
        let sa_inv = invert(&self.sbox[ga]);
        let sb_inv = invert(&self.sbox[gb]);
        let sa_new = draw_sbox(rng, wa);
        let sb_new = draw_sbox(rng, wb);
        let ma = (1usize << wa) - 1;
        let t: Perm = (0..(1usize << (wa + wb)))
            .map(|u| {
                let mut sa = sa_inv[u & ma] as usize;
                let mut sb = sb_inv[u >> wa] as usize;
                let (bit_a, bit_b) = ((sa >> ia) & 1, (sb >> ib) & 1);
                sa = (sa & !(1 << ia)) | (bit_b << ia);
                sb = (sb & !(1 << ib)) | (bit_a << ib);
                (sa_new[sa] as usize | ((sb_new[sb] as usize) << wa)) as u16
            })
            .collect();
        let ext = ext_pair(&t, wa, wb, self.nband, &self.mask[ga], &self.mask[gb]);
        let mc = synth(&ext, wa + wb + self.nband);
        let mut wires = self.groups[ga].clone();
        wires.extend(self.groups[gb].clone());
        wires.extend(self.band_wires.iter().copied());
        self.push(&wires, &mc);
        self.sbox[ga] = sa_new;
        self.sbox[gb] = sb_new;
        let (ca, cb) = (self.slot_coord[pa], self.slot_coord[pb]);
        self.slot_coord[pa] = cb;
        self.slot_coord[pb] = ca;
        self.coord_slot[ca] = pb;
        self.coord_slot[cb] = pa;
        self.n_surgery += 1;
    }

    fn inject(&mut self, gi: usize, sg: &SrcGate, rng: &mut Rng) {
        let w = self.gw(gi);
        let wires_grp = self.groups[gi].clone();
        let pos: HashMap<usize, usize> =
            wires_grp.iter().enumerate().map(|(j, &p)| (self.slot_coord[p], j)).collect();
        let tpos = pos[&sg.target];
        let cpos: Vec<(usize, bool)> = sg.ctrls.iter().map(|&(c, p)| (pos[&c], p)).collect();
        let s_inv = invert(&self.sbox[gi]);
        let s_new = draw_sbox(rng, w);
        let t: Perm = (0..(1usize << w))
            .map(|z| {
                let s = s_inv[z] as usize;
                let all = cpos.iter().all(|&(j, pol)| (((s >> j) & 1) == 1) == pol);
                let s2 = if sg.comp ^ all { s ^ (1 << tpos) } else { s };
                s_new[s2] as u16
            })
            .collect();
        let ext = ext_single(&t, w, self.nband, &self.mask[gi], true, true);
        let mc = synth(&ext, w + self.nband);
        let wires = self.single_wires(gi);
        self.push(&wires, &mc);
        self.sbox[gi] = s_new;
        self.n_inject += 1;
    }

    fn tail_decode(&mut self) {
        for gi in 0..self.groups.len() {
            let w = self.gw(gi);
            let t = invert(&self.sbox[gi]);
            // input masked, output raw: unmask, no remask.
            let ext = ext_single(&t, w, self.nband, &self.mask[gi], true, false);
            let mc = synth(&ext, w + self.nband);
            let wires = self.single_wires(gi);
            self.push(&wires, &mc);
            self.sbox[gi] = identity(w);
        }
        let mut cur = self.slot_coord.clone();
        for p in 0..self.n {
            while cur[p] != p {
                let q = cur[p];
                self.gates.push(XGate::cnot(p as u16, q as u16));
                self.gates.push(XGate::cnot(q as u16, p as u16));
                self.gates.push(XGate::cnot(p as u16, q as u16));
                cur.swap(p, q);
            }
        }
        self.slot_coord = (0..self.n).collect();
        self.coord_slot = (0..self.n).collect();
    }
}

// ---------------------------------------------------------------- verify

/// Run `src` (payload only) and `fab` (payload + band) on the same random zero
/// slice and compare ONLY the payload wires `0..n_payload`.
fn payload_agrees(
    src: &[XGate],
    fab: &[XGate],
    n_payload: usize,
    n_total: usize,
    blk: usize,
    reps: usize,
    rng: &mut Rng,
) -> bool {
    let run = |gates: &[XGate], st: &mut Vec<u64>| {
        for g in gates {
            let mut m = !0u64;
            for &(w, p) in &g.ctrls {
                m &= if p { st[w as usize] } else { !st[w as usize] };
            }
            if g.comp {
                m = !m;
            }
            st[g.target as usize] ^= m;
        }
    };
    for _ in 0..reps {
        let mut s_src = vec![0u64; n_total];
        for w in 0..blk {
            s_src[w] = rng.next();
        }
        let mut s_fab = s_src.clone();
        run(src, &mut s_src);
        run(fab, &mut s_fab);
        if s_src[0..n_payload] != s_fab[0..n_payload] {
            return false;
        }
    }
    true
}

// ---------------------------------------------------------------- main

fn main() {
    let a: Vec<String> = std::env::args().collect();
    let get = |k: &str| a.iter().position(|s| s == k).map(|i| a[i + 1].clone());

    let g: usize = get("--g").map(|s| s.parse().unwrap()).unwrap_or(3);
    let nband: usize = get("--nband").map(|s| s.parse().unwrap()).unwrap_or(4);
    let m_lane = get("--m-lane").unwrap_or_else(|| "prod".into());
    let prod = match m_lane.as_str() {
        "prod" => true,
        "idle" => false,
        other => panic!("--m-lane must be prod|idle (got {other})"),
    };
    let dummy: usize = get("--dummy").map(|s| s.parse().unwrap()).unwrap_or(0);
    let source = get("--source").expect("--source <sandwich.mpmct1>");
    let out = get("--out").expect("--out <p.mpmct1>");
    let verify_reps: usize = get("--verify").map(|s| s.parse().unwrap()).unwrap_or(4);
    let seed = get("--seed").map(|s| s.parse::<u64>().unwrap()).unwrap_or_else(|| {
        std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap()
            .subsec_nanos() as u64
            ^ ((std::process::id() as u64) << 32)
            ^ 0x5DEECE66D
    });
    let mut rng = Rng(seed);

    let t0 = Instant::now();
    let (sg, nw) = read_mpmct(&source).expect("read source");
    let limit: usize = get("--gates").map(|s| s.parse().unwrap()).unwrap_or(sg.len());
    let blk = nw / 2;
    let src: Vec<SrcGate> = sg
        .iter()
        .take(limit)
        .map(|x| SrcGate {
            target: x.target as usize,
            comp: x.comp,
            ctrls: x.ctrls.iter().map(|&(w, p)| (w as usize, p)).collect(),
        })
        .collect();
    let maxsup = src.iter().map(|s| s.support().len()).max().unwrap_or(0);
    let mut fab = Fabric::new(nw, g, nband, prod, blk, &mut rng);
    eprintln!(
        "source {} gates / {} payload wires (using {}), max support {}, {} groups (g={}{}), band {} (m-lane {})",
        sg.len(),
        nw,
        src.len(),
        maxsup,
        fab.groups.len(),
        g,
        if nw % g != 0 { ", remainder folded into the last" } else { "" },
        nband,
        m_lane,
    );
    assert!(maxsup <= g, "group size {g} cannot hold support {maxsup}; raise --g");

    fab.band_fill();
    let fill_gates = fab.gates.len();
    fab.head_encode(&mut rng);
    let head_gates = fab.gates.len() - fill_gates;

    for (i, s) in src.iter().enumerate() {
        let sup = s.support();
        let mut counts: HashMap<usize, usize> = HashMap::new();
        for &c in &sup {
            *counts.entry(fab.wire_group[fab.coord_slot[c]]).or_insert(0) += 1;
        }
        let home = fab.wire_group[fab.coord_slot[s.target]];
        let dest = *counts
            .iter()
            .max_by_key(|&(gi, n)| (*n, (*gi == home) as usize))
            .map(|(gi, _)| gi)
            .unwrap();
        for &c in &sup {
            if fab.wire_group[fab.coord_slot[c]] == dest {
                continue;
            }
            let victim = fab.groups[dest]
                .iter()
                .copied()
                .find(|&p| !sup.contains(&fab.slot_coord[p]))
                .expect("free slot in destination (support <= g)");
            let from = fab.coord_slot[c];
            fab.surgery(from, victim, &mut rng);
        }
        fab.inject(dest, s, &mut rng);
        for _ in 0..dummy {
            let gi = rng.below(fab.groups.len());
            fab.refresh(gi, &mut rng);
        }
        if i > 0 && i % 1000 == 0 {
            eprintln!(
                "  {i}/{} folds, {} gates, {:.1}s",
                src.len(),
                fab.gates.len(),
                t0.elapsed().as_secs_f64()
            );
        }
    }
    let body_gates = fab.gates.len() - fill_gates - head_gates;
    fab.tail_decode();

    let mut hist: HashMap<usize, usize> = HashMap::new();
    for x in &fab.gates {
        *hist.entry(x.ctrls.len()).or_insert(0) += 1;
    }
    let mut hv: Vec<_> = hist.into_iter().collect();
    hv.sort();
    let total = fab.gates.len();
    let folds = src.len().max(1);
    eprintln!("\n=== FASKRI + M-lane (g={g}, nband={nband}, m-lane {m_lane}) ===");
    eprintln!(
        "blocks   : inject {}  surgery {}  refresh {}",
        fab.n_inject, fab.n_surgery, fab.n_refresh
    );
    eprintln!(
        "gates    : {total} total  (fill {fill_gates}, head {head_gates}, body {body_gates}, tail {})",
        total - fill_gates - head_gates - body_gates
    );
    eprintln!(
        "           {:.1} gates/fold, {:.2} surgeries/fold",
        total as f64 / folds as f64,
        fab.n_surgery as f64 / folds as f64
    );
    eprintln!("control-count histogram (g57+CNOT digestible is <=2):");
    for (k, v) in &hv {
        eprintln!("    {k} controls : {v} ({:.1}%)", 100.0 * *v as f64 / total as f64);
    }
    let narrow: usize = hv.iter().filter(|(k, _)| *k <= 2).map(|(_, v)| *v).sum();
    eprintln!("    <=2 controls (directly digestible): {:.1}%", 100.0 * narrow as f64 / total as f64);

    if verify_reps > 0 {
        let src_gates: Vec<XGate> = sg.iter().take(limit).cloned().collect();
        let n_total = nw + nband;
        let ok = payload_agrees(&src_gates, &fab.gates, nw, n_total, blk, verify_reps, &mut rng);
        eprintln!(
            "functional verify vs source on {} zero-slice inputs (payload wires only): {}",
            verify_reps * 64,
            if ok { "PASS" } else { "FAIL" }
        );
        assert!(ok, "emitted circuit does not match the source permutation on payload");
    }
    write_mpmct(&out, &fab.gates, nw + nband).expect("write out");
    #[cfg(unix)]
    {
        use std::io::Write;
        use std::os::unix::fs::OpenOptionsExt;
        if let Ok(mut f) = std::fs::OpenOptions::new()
            .write(true)
            .create(true)
            .truncate(true)
            .mode(0o600)
            .open(format!("{out}.SEED"))
        {
            let _ = writeln!(f, "{seed}");
        }
    }
    eprintln!("wrote {out}  ({:.1}s)", t0.elapsed().as_secs_f64());
}
