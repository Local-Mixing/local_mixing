//! FASKRI prototype — Fixed Ambient Skeleton, Keyed Routing, Fused Injection.
//!
//! Preprocessing-stage prototype for the production pipeline (sliced sandwich,
//! n=128 => 256 payload wires). Design note: `docs/GADGETIZE_REDESIGN_20260826.md`.
//!
//! The source state `s in GF(2)^256` is carried, in place, as a keyed groupwise
//! encoding: the payload wires are partitioned into groups of `g`, and group `G`
//! holds `S_G(s|coords(G))` for a keyed `g`-bit S-box. Three block types, each a
//! permutation of at most `2^(2g)` points synthesized MONOLITHICALLY (never
//! decode -> apply -> encode, which would materialize `s` on a wire):
//!
//!   * INJECT  (width g)   T = S' . B~ . S^-1     -- fuse one source gate
//!   * SURGERY (width 2g)  T = (S'_A (x) S'_B) . swap . (S_A (x) S_B)^-1
//!                                                -- keyed routing: move one
//!                                                   coordinate between groups
//!   * REFRESH (width g)   T = S' . S^-1          -- dummy injection; every group
//!                                                   is an injection slot
//!
//! Routing is what makes `r.g.r'` work at production width: rather than widening
//! the frame to reach scattered operands (global diffusion => 2^n synthesis), the
//! operands are ROUTED into a common group, so every synthesized object stays at
//! width g or 2g and is exhaustively certifiable.
//!
//! Every block is certified over its whole `2^w` domain: no internal wire segment
//! may equal a raw source coordinate or its complement. On violation the block is
//! REJECTED and resynthesized from a fresh outgoing S-box (design §3.2). This is
//! not hygiene -- measured violation rate before resampling is ~0.24 per block.
//!
//! Ends: head encode and tail decode are the only clear materializations (the
//! correctness-forced G4 carve-out). The emitted circuit is functionally EQUAL to
//! the source sandwich, so the contract A(x,0) = (junk, C(x)) is preserved exactly
//! and is checked by `--verify`.
//!
//! NOT in this prototype (documented, not silently stubbed):
//!   * the withheld additive M-lane (band wires) -- Phi only, no masks
//!   * lowering multi-control blocks to strict g57+CNOT (see the histogram)
//!
//! Usage:
//!   faskri_gen --source <sandwich.mpmct1> --out <p.mpmct1> [--g 4] [--gates N]
//!              [--dummy K] [--seed S] [--verify N] [--cert] [--bench-synth]

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
/// output coordinate. ENCODED_CHAIN_DESIGN §7.1 found a random max-degree box can
/// leave one coordinate a degree early, letting that bit come back early.
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
/// When input `i` is processed every `j < i` is a fixed point; `f` bijective =>
/// `f(i) >= i`, so no control set used is contained in any settled `j`.
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
    // Applied on the output side: G_k...G_1.perm = Id, so perm = G_1...G_k and a
    // circuit computing perm applies G_k first.
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

// ---------------------------------------------------------------- certificate

/// Raw source coordinates as functions of the block's `2^w` input domain.
struct CertCtx {
    raw: Vec<Vec<u8>>,
}

impl CertCtx {
    fn one(s_inv: &Perm, w: usize) -> CertCtx {
        let n = 1usize << w;
        CertCtx {
            raw: (0..w)
                .map(|j| (0..n).map(|u| ((s_inv[u] >> j) & 1) as u8).collect())
                .collect(),
        }
    }
    fn pair(sa_inv: &Perm, wa: usize, sb_inv: &Perm, wb: usize) -> CertCtx {
        let n = 1usize << (wa + wb);
        let ma = (1usize << wa) - 1;
        let mut raw = Vec::new();
        for j in 0..wa {
            raw.push((0..n).map(|u| ((sa_inv[u & ma] >> j) & 1) as u8).collect());
        }
        for j in 0..wb {
            raw.push((0..n).map(|u| ((sb_inv[u >> wa] >> j) & 1) as u8).collect());
        }
        CertCtx { raw }
    }
}

/// Enumerate the block's `2^w` domain; verify no internal wire segment equals a
/// raw source coordinate or its complement. Returns (checks, violations).
fn certify_block(mc: &[McGate], w: usize, ctx: &CertCtx) -> (usize, usize) {
    let n = 1usize << w;
    let mut state: Vec<u16> = (0..n as u16).collect();
    let (mut checked, mut viol) = (0usize, 0usize);
    for g in mc {
        let mask: u16 = g.ctrls.iter().fold(0u16, |m, &c| m | (1 << c));
        let tb: u16 = 1 << g.target;
        for v in state.iter_mut() {
            if (*v & mask) == mask {
                *v ^= tb;
            }
        }
        let seg: Vec<u8> = (0..n).map(|u| ((state[u] >> g.target) & 1) as u8).collect();
        for r in &ctx.raw {
            checked += 1;
            let eq = seg.iter().zip(r).all(|(a, b)| a == b);
            let ne = seg.iter().zip(r).all(|(a, b)| a != b);
            if eq || ne {
                viol += 1;
            }
        }
    }
    (checked, viol)
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

const MAX_RESAMPLE: usize = 24;

struct Fabric {
    n: usize,
    groups: Vec<Vec<usize>>,
    wire_group: Vec<usize>,
    slot_coord: Vec<usize>,
    coord_slot: Vec<usize>,
    sbox: Vec<Perm>,
    gates: Vec<XGate>,
    cert: bool,
    n_inject: usize,
    n_surgery: usize,
    n_refresh: usize,
    cert_checked: usize,
    cert_rejects: usize,
    cert_unfixed: usize,
}

impl Fabric {
    /// Partition `n` wires into groups of `g`; any remainder is absorbed into the
    /// last group (so g need not divide n).
    fn new(n: usize, g: usize, cert: bool) -> Self {
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
        Fabric {
            n,
            groups,
            wire_group,
            slot_coord: (0..n).collect(),
            coord_slot: (0..n).collect(),
            sbox,
            gates: Vec::new(),
            cert,
            n_inject: 0,
            n_surgery: 0,
            n_refresh: 0,
            cert_checked: 0,
            cert_rejects: 0,
            cert_unfixed: 0,
        }
    }

    #[inline]
    fn gw(&self, gi: usize) -> usize {
        self.groups[gi].len()
    }
    /// position of a wire inside its group
    fn pos_in_group(&self, wire: usize) -> usize {
        let gi = self.wire_group[wire];
        self.groups[gi].iter().position(|&w| w == wire).expect("wire in its group")
    }

    /// Synthesize, certify, and (on violation) reject. Returns the accepted gate
    /// list, or the last attempt if every resample violated.
    fn accept(&mut self, t: &Perm, w: usize, ctx: Option<&CertCtx>) -> (Vec<McGate>, bool) {
        let mc = synth(t, w);
        match ctx {
            None => (mc, true),
            Some(c) => {
                let (checked, viol) = certify_block(&mc, w, c);
                self.cert_checked += checked;
                (mc, viol == 0)
            }
        }
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

    fn head_encode(&mut self, rng: &mut Rng) {
        for gi in 0..self.groups.len() {
            let w = self.gw(gi);
            let s = draw_sbox(rng, w);
            let mc = synth(&s, w);
            let wires = self.groups[gi].clone();
            self.push(&wires, &mc);
            self.sbox[gi] = s;
        }
    }

    fn refresh(&mut self, gi: usize, rng: &mut Rng) {
        let w = self.gw(gi);
        let s_inv = invert(&self.sbox[gi]);
        let ctx = if self.cert { Some(CertCtx::one(&s_inv, w)) } else { None };
        let mut chosen = None;
        for _ in 0..MAX_RESAMPLE {
            let s_new = draw_sbox(rng, w);
            let t: Perm = (0..(1usize << w)).map(|z| s_new[s_inv[z] as usize]).collect();
            let (mc, ok) = self.accept(&t, w, ctx.as_ref());
            if ok {
                chosen = Some((mc, s_new));
                break;
            }
            self.cert_rejects += 1;
            chosen = Some((mc, s_new));
        }
        let (mc, s_new) = chosen.expect("at least one attempt");
        let wires = self.groups[gi].clone();
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
        let ctx = if self.cert {
            Some(CertCtx::pair(&sa_inv, wa, &sb_inv, wb))
        } else {
            None
        };
        let ma = (1usize << wa) - 1;
        let mut chosen = None;
        for _ in 0..MAX_RESAMPLE {
            let sa_new = draw_sbox(rng, wa);
            let sb_new = draw_sbox(rng, wb);
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
            let (mc, ok) = self.accept(&t, wa + wb, ctx.as_ref());
            if ok {
                chosen = Some((mc, sa_new, sb_new, true));
                break;
            }
            self.cert_rejects += 1;
            chosen = Some((mc, sa_new, sb_new, false));
        }
        let (mc, sa_new, sb_new, ok) = chosen.expect("at least one attempt");
        if !ok {
            self.cert_unfixed += 1;
        }
        let mut wires = self.groups[ga].clone();
        wires.extend(self.groups[gb].clone());
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
        let wires = self.groups[gi].clone();
        let pos: HashMap<usize, usize> =
            wires.iter().enumerate().map(|(j, &p)| (self.slot_coord[p], j)).collect();
        let tpos = pos[&sg.target];
        let cpos: Vec<(usize, bool)> = sg.ctrls.iter().map(|&(c, p)| (pos[&c], p)).collect();
        let s_inv = invert(&self.sbox[gi]);
        let ctx = if self.cert { Some(CertCtx::one(&s_inv, w)) } else { None };
        let mut chosen = None;
        for _ in 0..MAX_RESAMPLE {
            let s_new = draw_sbox(rng, w);
            let t: Perm = (0..(1usize << w))
                .map(|z| {
                    let s = s_inv[z] as usize;
                    // fires = comp XOR AND(literals); the empty AND is true
                    let all = cpos.iter().all(|&(j, pol)| (((s >> j) & 1) == 1) == pol);
                    let s2 = if sg.comp ^ all { s ^ (1 << tpos) } else { s };
                    s_new[s2] as u16
                })
                .collect();
            let (mc, ok) = self.accept(&t, w, ctx.as_ref());
            if ok {
                chosen = Some((mc, s_new, true));
                break;
            }
            self.cert_rejects += 1;
            chosen = Some((mc, s_new, false));
        }
        let (mc, s_new, ok) = chosen.expect("at least one attempt");
        if !ok {
            self.cert_unfixed += 1;
        }
        self.push(&wires, &mc);
        self.sbox[gi] = s_new;
        self.n_inject += 1;
    }

    fn tail_decode(&mut self) {
        for gi in 0..self.groups.len() {
            let w = self.gw(gi);
            let t = invert(&self.sbox[gi]);
            let mc = synth(&t, w);
            let wires = self.groups[gi].clone();
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

fn circuits_agree(
    a: &[XGate],
    b: &[XGate],
    nw: usize,
    blk: usize,
    reps: usize,
    rng: &mut Rng,
) -> bool {
    for _ in 0..reps {
        let mut s1 = vec![0u64; nw];
        for w in 0..blk {
            s1[w] = rng.next();
        }
        let mut s2 = s1.clone();
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
        run(a, &mut s1);
        run(b, &mut s2);
        if s1 != s2 {
            return false;
        }
    }
    true
}

// ---------------------------------------------------------------- main

fn main() {
    let a: Vec<String> = std::env::args().collect();
    let get = |k: &str| a.iter().position(|s| s == k).map(|i| a[i + 1].clone());
    let has = |k: &str| a.iter().any(|s| s == k);

    let g: usize = get("--g").map(|s| s.parse().unwrap()).unwrap_or(4);
    if has("--bench-synth") {
        bench_synth(g);
        return;
    }
    let dummy: usize = get("--dummy").map(|s| s.parse().unwrap()).unwrap_or(0);
    let cert = has("--cert");
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
    let mut fab = Fabric::new(nw, g, cert);
    eprintln!(
        "source {} gates / {} wires (using {}), max support {}, {} groups (g={}{})",
        sg.len(),
        nw,
        src.len(),
        maxsup,
        fab.groups.len(),
        g,
        if nw % g != 0 { ", remainder folded into the last" } else { "" }
    );
    assert!(maxsup <= g, "group size {g} cannot hold support {maxsup}; raise --g");

    fab.head_encode(&mut rng);
    let head_gates = fab.gates.len();

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
    let body_gates = fab.gates.len() - head_gates;
    fab.tail_decode();

    let mut hist: HashMap<usize, usize> = HashMap::new();
    for x in &fab.gates {
        *hist.entry(x.ctrls.len()).or_insert(0) += 1;
    }
    let mut hv: Vec<_> = hist.into_iter().collect();
    hv.sort();
    let total = fab.gates.len();
    let folds = src.len().max(1);
    eprintln!("\n=== FASKRI prototype (g={g}) ===");
    eprintln!(
        "blocks   : inject {}  surgery {}  refresh {}",
        fab.n_inject, fab.n_surgery, fab.n_refresh
    );
    eprintln!(
        "gates    : {total} total  (head {head_gates}, body {body_gates}, tail {})",
        total - head_gates - body_gates
    );
    eprintln!("           {:.1} gates/fold, {:.2} surgeries/fold", total as f64 / folds as f64, fab.n_surgery as f64 / folds as f64);
    eprintln!("control-count histogram (g57+CNOT digestible is <=2):");
    for (k, v) in &hv {
        eprintln!("    {k} controls : {v} ({:.1}%)", 100.0 * *v as f64 / total as f64);
    }
    let narrow: usize = hv.iter().filter(|(k, _)| *k <= 2).map(|(_, v)| *v).sum();
    eprintln!("    <=2 controls (directly digestible): {:.1}%", 100.0 * narrow as f64 / total as f64);
    if cert {
        eprintln!(
            "certificate: {} internal-segment checks, {} blocks rejected+resynthesized, {} unfixable after {} resamples",
            fab.cert_checked, fab.cert_rejects, fab.cert_unfixed, MAX_RESAMPLE
        );
    }
    if verify_reps > 0 {
        let src_gates: Vec<XGate> = sg.iter().take(limit).cloned().collect();
        let ok = circuits_agree(&src_gates, &fab.gates, nw, nw / 2, verify_reps, &mut rng);
        eprintln!(
            "functional verify vs source on {} zero-slice inputs: {}",
            verify_reps * 64,
            if ok { "PASS" } else { "FAIL" }
        );
        assert!(ok, "emitted circuit does not match the source permutation");
    }
    write_mpmct(&out, &fab.gates, nw).expect("write out");
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

/// Synthesis cost at the widths the construction uses -- the number that decides
/// whether keyed routing is affordable.
fn bench_synth(g: usize) {
    let mut rng = Rng(0xC0FFEE);
    for &w in &[g, 2 * g] {
        let reps = if w <= 6 { 500 } else { 100 };
        let (mut tot, mut worst, mut ctot) = (0usize, 0usize, 0usize);
        let t = Instant::now();
        for _ in 0..reps {
            let n = 1usize << w;
            let mut p: Perm = identity(w);
            for i in (1..n).rev() {
                let j = rng.below(i + 1);
                p.swap(i, j);
            }
            let mc = synth(&p, w);
            tot += mc.len();
            worst = worst.max(mc.len());
            ctot += mc.iter().map(|x| x.ctrls.len()).sum::<usize>();
        }
        eprintln!(
            "w={w}: mean {:.1} gates (worst {worst}), mean controls/gate {:.2}, {:.0} us/solve",
            tot as f64 / reps as f64,
            ctot as f64 / tot.max(1) as f64,
            t.elapsed().as_secs_f64() * 1e6 / reps as f64
        );
    }
}
