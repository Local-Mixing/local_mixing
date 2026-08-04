// The swap-word engine: online synthesis of all-g57 realizations for the
// twist bracket (--twist-g57).
//
// The problem it answers: at a bracket seam the placer holds a few real
// neighborhood gates H and wants the SHORTEST all-g57 word R with
//
//     R == H . S_ab        (opening seam; the closing seam is S_ab . H)
//
// so the bracket is spelled in bulk material and pays |R| - |H| net gates
// instead of a fixed packet. An offline per-context table (the
// g57_swap_identities sqlite enumeration) keys this on SYNTAX and needs 218k
// synthesized rows; but the answer only depends on the PERMUTATION of H . S,
// so two target-independent BFS tables over the 24 g57 gates on 4 abstract
// wires answer every context — any k, any cut, 3-wire supports, non-g57
// neighbors — by one meet-in-the-middle lookup. A 16-state permutation packs
// into one u64 (nibble s = image of state s); the radius-4 ball is ~165k
// perms and builds in milliseconds.
//
// Measured ground truth this engine reproduces (exhaustive, 2026-07-29):
//   dist(S_ab) = 6 over the 24-gate alphabet (the 4th wire buys nothing);
//   every k=2 context in the sqlite DB's 4-wire scope sits at distance 6
//   (net +4), while a pair on a SINGLE 3-wire support always admits length 4
//   (net +2) — the placer prefers small supports for exactly that reason.
use super::xgate::XGate;
use std::collections::HashMap;
use std::hash::{BuildHasherDefault, Hasher};

/// Keys are already well-mixed 64-bit permutation encodings; a Fibonacci
/// multiply beats SipHash by several-fold on the hot MITM probe loop.
#[derive(Default)]
pub struct PermHasher(u64);

impl Hasher for PermHasher {
    #[inline]
    fn write(&mut self, _: &[u8]) {
        unreachable!("PermHasher only hashes u64 keys");
    }
    #[inline]
    fn write_u64(&mut self, x: u64) {
        self.0 = x.wrapping_mul(0x9E37_79B9_7F4A_7C15);
    }
    #[inline]
    fn finish(&self) -> u64 {
        self.0
    }
}

type PermMap<V> = HashMap<u64, V, BuildHasherDefault<PermHasher>>;

/// Abstract wires 0..3 (a, b, c, d); the distinguished swap acts on (0, 1).
pub const ABS_WIRES: usize = 4;
const NSTATES: usize = 16;
pub const NGATES: usize = 24;

/// The longest word solve() can return: front (<=3) + back (<=4).
pub const MAX_WORD: usize = 7;

/// Identity permutation, packed nibble-per-state.
const fn ident() -> u64 {
    let mut p = 0u64;
    let mut s = 0;
    while s < NSTATES {
        p |= (s as u64) << (4 * s);
        s += 1;
    }
    p
}
pub const IDENT: u64 = ident();

#[inline(always)]
fn nib(p: u64, i: usize) -> usize {
    ((p >> (4 * i)) & 0xF) as usize
}

/// Apply `f` then `g`.
#[inline]
pub fn compose(f: u64, g: u64) -> u64 {
    let mut out = 0u64;
    for s in 0..NSTATES {
        out |= (nib(g, nib(f, s)) as u64) << (4 * s);
    }
    out
}

pub fn invert(f: u64) -> u64 {
    let mut out = 0u64;
    for s in 0..NSTATES {
        out |= (s as u64) << (4 * nib(f, s));
    }
    out
}

/// The pin assignment of gate index i: (target, negative ctrl, positive ctrl),
/// enumeration order fixed by construction and relied on nowhere else.
pub fn gate_pins(i: usize) -> (u8, u8, u8) {
    debug_assert!(i < NGATES);
    let mut idx = 0;
    for t in 0..ABS_WIRES as u8 {
        for n in 0..ABS_WIRES as u8 {
            if n == t {
                continue;
            }
            for p in 0..ABS_WIRES as u8 {
                if p == t || p == n {
                    continue;
                }
                if idx == i {
                    return (t, n, p);
                }
                idx += 1;
            }
        }
    }
    unreachable!()
}

/// perm of the g57 [t,n,p]: t ^= 1 XOR (p AND NOT n)  (fires = n OR NOT p).
fn g57_perm(t: u8, n: u8, p: u8) -> u64 {
    let mut out = 0u64;
    for s in 0..NSTATES {
        let fires = 1 ^ (((s >> p) & 1) & (1 - ((s >> n) & 1)));
        out |= ((s ^ (fires << t)) as u64) << (4 * s);
    }
    out
}

/// perm of an arbitrary XGate already mapped onto the abstract wires.
/// fires(x) = comp XOR AND_i lit_i, effect x[target] ^= fires.
pub fn xgate_perm(target: u8, ctrls: &[(u8, bool)], comp: bool) -> u64 {
    let mut out = 0u64;
    for s in 0..NSTATES {
        let mut m = 1usize;
        for &(w, pol) in ctrls {
            let bit = (s >> w) & 1;
            m &= if pol { bit } else { 1 - bit };
        }
        let fires = (comp as usize) ^ m;
        out |= ((s ^ (fires << target)) as u64) << (4 * s);
    }
    out
}

/// The distinguished swap: exchange abstract wires 0 and 1.
fn swap_ab() -> u64 {
    let mut out = 0u64;
    for s in 0..NSTATES {
        let r = (s & !3) | ((s & 1) << 1) | ((s >> 1) & 1);
        out |= (r as u64) << (4 * s);
    }
    out
}

/// A word of gate indices, packed for the back table.
#[derive(Clone, Copy)]
struct BackWord {
    len: u8,
    gates: [u8; 4],
}

struct FrontEntry {
    len: u8,
    inv: u64,
    gates: [u8; 3],
}

pub struct SwapWordEngine {
    gate_perms: [u64; NGATES],
    pub s_ab: u64,
    /// perm -> shortest word of length <= 4 realizing it (apply order).
    back: PermMap<BackWord>,
    /// All shortest words of length <= 3, sorted by length ascending.
    front: Vec<FrontEntry>,
    /// The context-free bracket: a fixed shortest word for S_ab itself, so
    /// the k = 0 seam never pays a MITM scan.
    bare: Vec<u8>,
    pub build_ms: u64,
    pub back_len: usize,
}

impl SwapWordEngine {
    pub fn new() -> SwapWordEngine {
        let t0 = std::time::Instant::now();
        let mut gate_perms = [0u64; NGATES];
        for (i, gp) in gate_perms.iter_mut().enumerate() {
            let (t, n, p) = gate_pins(i);
            *gp = g57_perm(t, n, p);
        }
        // BFS to radius 4 keeping the first (shortest) word per perm. A
        // shortest word never repeats a gate adjacently (g57s are
        // involutions), so that branch is pruned.
        let mut back: PermMap<BackWord> =
            PermMap::with_capacity_and_hasher(200_000, Default::default());
        back.insert(
            IDENT,
            BackWord {
                len: 0,
                gates: [0; 4],
            },
        );
        let mut frontier: Vec<(u64, BackWord)> = vec![(
            IDENT,
            BackWord {
                len: 0,
                gates: [0; 4],
            },
        )];
        for depth in 1..=4u8 {
            let mut next: Vec<(u64, BackWord)> = Vec::with_capacity(frontier.len() * 20);
            for &(perm, w) in &frontier {
                for (gi, gp) in gate_perms.iter().enumerate() {
                    if w.len > 0 && w.gates[w.len as usize - 1] == gi as u8 {
                        continue;
                    }
                    let q = compose(perm, *gp);
                    if let std::collections::hash_map::Entry::Vacant(e) = back.entry(q) {
                        let mut nw = w;
                        nw.gates[nw.len as usize] = gi as u8;
                        nw.len = depth;
                        e.insert(nw);
                        next.push((q, nw));
                    }
                }
            }
            frontier = next;
        }
        // Front list: the radius-3 subset, inverse perms precomputed, sorted
        // by length so solve() can stop as soon as no later entry can win.
        let mut front: Vec<FrontEntry> = back
            .iter()
            .filter(|(_, w)| w.len <= 3)
            .map(|(&perm, w)| FrontEntry {
                len: w.len,
                inv: invert(perm),
                gates: [w.gates[0], w.gates[1], w.gates[2]],
            })
            .collect();
        front.sort_by_key(|e| e.len);
        let mut engine = SwapWordEngine {
            gate_perms,
            s_ab: swap_ab(),
            back_len: back.len(),
            back,
            front,
            bare: Vec::new(),
            build_ms: t0.elapsed().as_millis() as u64,
        };
        engine.bare = engine
            .solve(engine.s_ab, MAX_WORD)
            .expect("S_ab is reachable");
        assert_eq!(
            engine.bare.len(),
            6,
            "dist(S_ab) must be 6 over the 24-gate alphabet"
        );
        engine
    }

    /// The fixed context-free spelling of S_ab (length 6).
    pub fn bare_word(&self) -> &[u8] {
        &self.bare
    }

    #[inline]
    pub fn gate_perm(&self, i: usize) -> u64 {
        self.gate_perms[i]
    }

    /// Shortest all-g57 word (gate indices, apply order) realizing `target`,
    /// of length <= maxlen; None when the ball does not reach it.
    pub fn solve(&self, target: u64, maxlen: usize) -> Option<Vec<u8>> {
        let mut best: Option<Vec<u8>> = None;
        for f in &self.front {
            if let Some(b) = &best {
                if b.len() <= f.len as usize {
                    break; // fronts only get longer from here
                }
            }
            // need = apply f.inv then target: the back word that completes f.
            let need = compose(f.inv, target);
            if let Some(bw) = self.back.get(&need) {
                let total = f.len as usize + bw.len as usize;
                if total <= maxlen && best.as_ref().map_or(true, |b| total < b.len()) {
                    let mut w = Vec::with_capacity(total);
                    w.extend_from_slice(&f.gates[..f.len as usize]);
                    w.extend_from_slice(&bw.gates[..bw.len as usize]);
                    best = Some(w);
                }
            }
        }
        best
    }

    /// Decode a solved word into physical XGates through `wires[abstract]`.
    pub fn decode(&self, word: &[u8], wires: &[u16; ABS_WIRES]) -> Vec<XGate> {
        word.iter()
            .map(|&gi| {
                let (t, n, p) = gate_pins(gi as usize);
                XGate::from_g57([wires[t as usize], wires[n as usize], wires[p as usize]])
            })
            .collect()
    }
}

/// Process-wide engine: deterministic, read-only after construction, ~ms to
/// build, so a single lazy instance serves every Mixer (and every test).
static ENGINE: std::sync::OnceLock<SwapWordEngine> = std::sync::OnceLock::new();

pub fn engine() -> &'static SwapWordEngine {
    ENGINE.get_or_init(SwapWordEngine::new)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn word_perm(e: &SwapWordEngine, w: &[u8]) -> u64 {
        w.iter()
            .fold(IDENT, |p, &g| compose(p, e.gate_perm(g as usize)))
    }

    #[test]
    fn engine_dist_s_ab_is_6() {
        let e = SwapWordEngine::new();
        let w = e.solve(e.s_ab, MAX_WORD).expect("swap solvable");
        assert_eq!(w.len(), 6);
        assert_eq!(word_perm(&e, &w), e.s_ab);
        // A context can realize S_ab exactly (e.g. a 3-CNOT swap), making the
        // seam target the identity: the empty word is the answer, and the
        // caller's net (0 - k) goes NEGATIVE — a shrinking seam, which the
        // selection arithmetic must represent (it once panicked on usize).
        assert_eq!(e.solve(IDENT, MAX_WORD).as_deref(), Some(&[][..]));
    }

    #[test]
    fn engine_reproduces_hidden_swap_identity() {
        // [a,b,c] . swap(a,b) . [b,c,a] == some 4-gate all-g57 word (the
        // HIDDEN_SWAP_IDENTITY in mix.rs exhibits one; here the engine must
        // find one of the same length).
        let e = SwapWordEngine::new();
        let g1 = xgate_perm(0, &[(1, false), (2, true)], true); // [a,b,c]
        let g2 = xgate_perm(1, &[(2, false), (0, true)], true); // [b,c,a]
        let target = compose(compose(g1, e.s_ab), g2);
        let w = e.solve(target, MAX_WORD).expect("pair context solvable");
        assert_eq!(w.len(), 4, "same-3-wire pair must cost net +2");
        assert_eq!(word_perm(&e, &w), target);
    }

    #[test]
    fn engine_solutions_verify_and_match_census() {
        // Cross-check against the exhaustive Python census (2026-07-29):
        // over ALL ordered pairs (h1, h2) of distinct 3-wire g57s on {a,b,c},
        // h1 . h2 . S_ab is solvable, and the length-4 count is 12 of 30
        // (fixed-op 2-prefix coverage 12/36 minus the 6 identical pairs).
        let e = SwapWordEngine::new();
        let mut on3: Vec<u64> = Vec::new();
        for i in 0..NGATES {
            let (t, n, p) = gate_pins(i);
            if t < 3 && n < 3 && p < 3 {
                on3.push(e.gate_perm(i));
            }
        }
        assert_eq!(on3.len(), 6);
        let (mut len4, mut seen) = (0, 0);
        for &h1 in &on3 {
            for &h2 in &on3 {
                if h1 == h2 {
                    continue;
                }
                seen += 1;
                let target = compose(compose(h1, h2), e.s_ab);
                // dist(h1.h2.S) ranges over [4, 8]; 8 exceeds the 3+4 MITM
                // reach and solve correctly returns None there — the placer
                // then consumes fewer gates instead.
                let Some(w) = e.solve(target, MAX_WORD) else {
                    continue;
                };
                assert_eq!(word_perm(&e, &w), target);
                assert!(w.len() >= 4 && w.len() <= 7);
                if w.len() == 4 {
                    len4 += 1;
                }
            }
        }
        assert_eq!(seen, 30);
        assert_eq!(
            len4, 12,
            "census: 12/30 ordered pairs at length 4 for fixed S_ab"
        );
    }

    #[test]
    fn engine_decode_roundtrip() {
        let e = SwapWordEngine::new();
        let w = e.solve(e.s_ab, MAX_WORD).unwrap();
        let gates = e.decode(&w, &[7, 3, 11, 0]);
        assert_eq!(gates.len(), 6);
        for g in &gates {
            assert!(g.comp && g.ctrls.len() == 2, "decoded gates are g57-shaped");
        }
    }
}
