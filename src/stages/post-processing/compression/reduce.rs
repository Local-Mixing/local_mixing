//! Zero-slice specialization, liveness and deterministic catalogue/ESOP reduction.
use super::*;

// Input-side specialization to the promised zero slice: with zero_in[w] wires
// known 0 at entry, run three-valued constant tracking (0 / 1 / unknown)
// forward once. A literal on a known wire either always holds (drop the
// literal) or never (the cube is dead: t ^= comp XOR 0, so the gate drops
// when comp=0 and degrades to a bare X when comp=1); a gate folded to an
// empty cube is an X (flips a known constant, stays) or a no-op comp gate
// (drops); any surviving write forgets its target's constant. The result
// equals the input circuit on every wire, but ONLY for entries inside the
// promised subspace.
pub(super) fn zero_specialize_anc(
    gates: Vec<XGate>,
    anc: Option<Vec<AncBits>>,
    zero_in: &[bool],
    wires: usize,
) -> (Vec<XGate>, Option<Vec<AncBits>>, usize, u64) {
    let mut known: Vec<Option<bool>> = (0..wires)
        .map(|w| {
            if zero_in.get(w) == Some(&true) {
                Some(false)
            } else {
                None
            }
        })
        .collect();
    let n = gates.len();
    let had_anc = anc.is_some();
    let mut out: Vec<XGate> = Vec::with_capacity(n);
    let mut out_anc: Option<Vec<AncBits>> = anc.as_ref().map(|_| Vec::with_capacity(n));
    let mut killed = 0usize;
    let mut lits_dropped = 0u64;
    let anc_iter = anc.unwrap_or_default();
    let mut anc_it = anc_iter.into_iter();
    for g in gates {
        let tag = if had_anc { anc_it.next() } else { None };
        let mut dead = false;
        let mut lits: Lits = Lits::new();
        for &(w, p) in &g.ctrls {
            match known[w as usize] {
                Some(v) if v == p => {} // literal always true: fold it away
                Some(_) => {
                    dead = true; // literal always false: the cube never fires
                    break;
                }
                None => lits.push((w, p)),
            }
        }
        let t = g.target as usize;
        if dead {
            // t ^= comp XOR 0: gone for comp=0, a bare NOT for comp=1.
            if g.comp {
                lits_dropped += g.width() as u64;
                known[t] = known[t].map(|v| !v);
                out.push(XGate::x_gate(g.target));
                if let Some(oa) = out_anc.as_mut() {
                    oa.push(tag.expect("sidecar aligned with gates"));
                }
            } else {
                killed += 1;
            }
            continue;
        }
        if lits.is_empty() {
            // t ^= comp XOR 1: an X gate, or a never-firing comp gate.
            if g.comp {
                killed += 1;
                continue;
            }
            lits_dropped += g.width() as u64;
            known[t] = known[t].map(|v| !v);
            out.push(XGate {
                target: g.target,
                comp: false,
                ctrls: lits,
            });
        } else {
            known[t] = None;
            lits_dropped += (g.width() - lits.len()) as u64;
            out.push(XGate {
                target: g.target,
                comp: g.comp,
                ctrls: lits,
            });
        }
        if let Some(oa) = out_anc.as_mut() {
            oa.push(tag.expect("sidecar aligned with gates"));
        }
    }
    (out, out_anc, killed, lits_dropped)
}

pub(super) fn liveness_prune_anc(
    gates: Vec<XGate>,
    anc: Option<Vec<AncBits>>,
    live_out: &[bool],
) -> (Vec<XGate>, Option<Vec<AncBits>>, usize) {
    let n = gates.len();
    let mut live = live_out.to_vec();
    let mut keep = vec![false; n];
    for i in (0..n).rev() {
        let g = &gates[i];
        if live[g.target as usize] {
            keep[i] = true;
            for &(w, _) in &g.ctrls {
                live[w as usize] = true;
            }
        }
    }
    let dropped = keep.iter().filter(|&&k| !k).count();
    let out: Vec<XGate> = gates
        .into_iter()
        .zip(keep.iter())
        .filter(|(_, k)| **k)
        .map(|(g, _)| g)
        .collect();
    let anc_out = anc.map(|a| {
        a.into_iter()
            .zip(keep.iter())
            .filter(|(_, k)| **k)
            .map(|(s, _)| s)
            .collect()
    });
    (out, anc_out, dropped)
}

// ---- exact minimum ESOP for supports of at most 4 wires -------------------
//
// One table per support size n in 1..=4. State = the function's truth table
// (2^n bits); edges = XOR one of the 3^n mixed-polarity cubes over exactly
// those n vars (the all-absent cube is the constant 1). BFS from 0 gives, for
// every function, a minimum-cube ESOP with a witness via parent pointers.
// Built lazily once per process; the largest (n=4) is 65536 states x 81 cubes.
struct ExactTab {
    from: Vec<u16>,       // parent state on a shortest path to 0
    via: Vec<u8>,         // cube index applied on that step
    cubes: Vec<(u8, u8)>, // (pos, neg) variable masks
}

static EXACT_TABS: OnceLock<[ExactTab; 4]> = OnceLock::new();

fn exact_tab_build(n: usize) -> ExactTab {
    let nbits = 1usize << n;
    let states = 1usize << nbits;
    let vmask = (nbits - 1) as u8; // n low variable bits
    let mut cubes: Vec<(u8, u8)> = Vec::new();
    for pos in 0u8..=vmask {
        for neg in 0u8..=vmask {
            if pos & neg == 0 {
                cubes.push((pos, neg));
            }
        }
    }
    let tts: Vec<u16> = cubes
        .iter()
        .map(|&(pos, neg)| {
            let mut tt: u16 = 0;
            for a in 0..nbits as u16 {
                if a as u8 & pos == pos && a as u8 & neg == 0 {
                    tt |= 1 << a;
                }
            }
            tt
        })
        .collect();
    let mut seen = vec![false; states];
    let mut from = vec![0u16; states];
    let mut via = vec![0u8; states];
    seen[0] = true;
    let mut q = VecDeque::from([0u16]);
    while let Some(s) = q.pop_front() {
        for (ci, &tt) in tts.iter().enumerate() {
            let t = s ^ tt;
            if !seen[t as usize] {
                seen[t as usize] = true;
                from[t as usize] = s;
                via[t as usize] = ci as u8;
                q.push_back(t);
            }
        }
    }
    debug_assert!(seen.iter().all(|&b| b), "minterm cubes reach every state");
    ExactTab { from, via, cubes }
}

// Minimum ESOP of the monomial set over n<=4 support vars: cubes plus a
// parity flip when the witness uses the constant cube.
pub(super) fn exact_small(monos: &[u64], n: usize) -> (Vec<(u64, u64)>, bool) {
    debug_assert!((1..=4).contains(&n));
    let tabs = EXACT_TABS.get_or_init(|| {
        [
            exact_tab_build(1),
            exact_tab_build(2),
            exact_tab_build(3),
            exact_tab_build(4),
        ]
    });
    let tab = &tabs[n - 1];
    let nbits = 1u16 << n;
    let mut f: u16 = 0;
    for &m in monos {
        let m = m as u16;
        for a in 0..nbits {
            if a & m == m {
                f ^= 1 << a;
            }
        }
    }
    let mut cubes: Vec<(u64, u64)> = Vec::new();
    let mut delta = false;
    let mut s = f;
    while s != 0 {
        let (pos, neg) = tab.cubes[tab.via[s as usize] as usize];
        if pos == 0 && neg == 0 {
            delta ^= true;
        } else {
            cubes.push((pos as u64, neg as u64));
        }
        s = tab.from[s as usize];
    }
    (cubes, delta)
}

// ---- maximum distance-1 matching ------------------------------------------
//
// Monomials differing in exactly one variable pair into a single-negation
// cube: mono(m) XOR mono(m|b) = mono(m) AND NOT b. The pair graph is
// bipartite by popcount parity, so Hopcroft-Karp finds a maximum matching
// (the old greedy pairing is kept as the fallback for oversized sets).
// Returns (cubes from matched pairs, unmatched monomials).
const MATCH_CAP: usize = 1 << 15;

pub(super) fn match_pairs(monos: &[u64], nbits: usize) -> (Vec<(u64, u64)>, Vec<u64>) {
    if monos.len() > MATCH_CAP {
        return greedy_pairs(monos);
    }
    let idx: HashMap<u64, usize> = monos.iter().enumerate().map(|(i, &m)| (m, i)).collect();
    let lefts: Vec<usize> = (0..monos.len())
        .filter(|&i| monos[i].count_ones() % 2 == 0)
        .collect();
    let adj: Vec<Vec<usize>> = lefts
        .iter()
        .map(|&i| {
            (0..nbits)
                .filter_map(|b| idx.get(&(monos[i] ^ (1u64 << b))).copied())
                .collect()
        })
        .collect();
    let mut pair_l: Vec<Option<usize>> = vec![None; lefts.len()]; // left pos -> mono idx
    let mut pair_r: Vec<Option<usize>> = vec![None; monos.len()]; // mono idx -> left pos
    loop {
        // BFS layers from free left vertices.
        let mut dist: Vec<Option<u32>> = vec![None; lefts.len()];
        let mut q: VecDeque<usize> = VecDeque::new();
        for (li, p) in pair_l.iter().enumerate() {
            if p.is_none() {
                dist[li] = Some(0);
                q.push_back(li);
            }
        }
        let mut found = false;
        while let Some(li) = q.pop_front() {
            for &v in &adj[li] {
                match pair_r[v] {
                    None => found = true,
                    Some(lj) => {
                        if dist[lj].is_none() {
                            dist[lj] = Some(dist[li].expect("queued has dist") + 1);
                            q.push_back(lj);
                        }
                    }
                }
            }
        }
        if !found {
            break;
        }
        fn dfs(
            li: usize,
            adj: &[Vec<usize>],
            dist: &mut [Option<u32>],
            pair_l: &mut [Option<usize>],
            pair_r: &mut [Option<usize>],
        ) -> bool {
            let d = dist[li];
            for vi in 0..adj[li].len() {
                let v = adj[li][vi];
                let ok = match pair_r[v] {
                    None => true,
                    Some(lj) => dist[lj] == d.map(|x| x + 1) && dfs(lj, adj, dist, pair_l, pair_r),
                };
                if ok {
                    pair_l[li] = Some(v);
                    pair_r[v] = Some(li);
                    return true;
                }
            }
            dist[li] = None;
            false
        }
        for li in 0..lefts.len() {
            if pair_l[li].is_none() {
                dfs(li, &adj, &mut dist, &mut pair_l, &mut pair_r);
            }
        }
    }
    let mut used = vec![false; monos.len()];
    let mut cubes = Vec::new();
    for (li, p) in pair_l.iter().enumerate() {
        if let Some(v) = *p {
            let (a, b) = (monos[lefts[li]], monos[v]);
            used[lefts[li]] = true;
            used[v] = true;
            cubes.push((a & b, a ^ b));
        }
    }
    let rest: Vec<u64> = (0..monos.len())
        .filter(|&i| !used[i])
        .map(|i| monos[i])
        .collect();
    (cubes, rest)
}

// The pre-2026-08 greedy pairing (pair m with m minus one bit), as the
// fallback when the monomial set is too large for Hopcroft-Karp.
pub(super) fn greedy_pairs(monos: &[u64]) -> (Vec<(u64, u64)>, Vec<u64>) {
    let mut order: Vec<u64> = monos.to_vec();
    order.sort_unstable_by_key(|&m| std::cmp::Reverse((m.count_ones(), m)));
    let pos_of: HashMap<u64, usize> = order.iter().enumerate().map(|(i, &m)| (m, i)).collect();
    let mut matched = vec![false; order.len()];
    let mut cubes = Vec::new();
    let mut rest = Vec::new();
    for i in 0..order.len() {
        if matched[i] {
            continue;
        }
        let m = order[i];
        let mut bits = m;
        let mut paired = false;
        while bits != 0 {
            let b = bits & bits.wrapping_neg();
            bits &= bits - 1;
            if let Some(&j) = pos_of.get(&(m & !b)) {
                if !matched[j] && j != i {
                    matched[i] = true;
                    matched[j] = true;
                    cubes.push((m & !b, b));
                    paired = true;
                    break;
                }
            }
        }
        if !paired {
            matched[i] = true;
            rest.push(m);
        }
    }
    (cubes, rest)
}

// ---- greedy subcube covering ----------------------------------------------
//
// A cube with negative-literal mask N replaces the 2^|N| monomials
// {pos|S : S subset of N} in one gate, so hunting for fully-present subcubes
// of dimension >= 2 beats any pairing. Best-first (largest dimension each
// round) up to COVER_EXHAUSTIVE_CAP monomials, single ascending-popcount pass
// beyond. Returns (cover cubes, residual monomials in input order).
const COVER_EXHAUSTIVE_CAP: usize = 1024;

pub(super) fn cover_grow(m: u64, set: &HashSet<u64>, nbits: usize) -> (u64, Vec<u64>) {
    let mut nmask = 0u64;
    let mut exp = vec![m];
    for b in 0..nbits {
        let bit = 1u64 << b;
        if m & bit != 0 || nmask & bit != 0 {
            continue;
        }
        if exp.iter().all(|&e| set.contains(&(e | bit))) {
            let add: Vec<u64> = exp.iter().map(|&e| e | bit).collect();
            exp.extend(add);
            nmask |= bit;
        }
    }
    (nmask, exp)
}

pub(super) fn greedy_cover(monos: &[u64], nbits: usize) -> (Vec<(u64, u64)>, Vec<u64>) {
    let mut set: HashSet<u64> = monos.iter().copied().collect();
    let mut cubes: Vec<(u64, u64)> = Vec::new();
    if monos.len() <= COVER_EXHAUSTIVE_CAP {
        loop {
            let mut best: Option<(u32, u64, u64, Vec<u64>)> = None;
            for &m in monos {
                if !set.contains(&m) {
                    continue;
                }
                let (nmask, exp) = cover_grow(m, &set, nbits);
                let dim = nmask.count_ones();
                if dim >= 2 && best.as_ref().is_none_or(|b| dim > b.0) {
                    best = Some((dim, m, nmask, exp));
                }
            }
            let Some((_, base, nmask, exp)) = best else {
                break;
            };
            cubes.push((base, nmask));
            for e in exp {
                set.remove(&e);
            }
        }
    } else {
        let mut by_pop: Vec<u64> = monos.to_vec();
        by_pop.sort_unstable_by_key(|&m| (m.count_ones(), m));
        for m in by_pop {
            if !set.contains(&m) {
                continue;
            }
            let (nmask, exp) = cover_grow(m, &set, nbits);
            if nmask.count_ones() >= 2 {
                cubes.push((m, nmask));
                for e in exp {
                    set.remove(&e);
                }
            }
        }
    }
    let residual: Vec<u64> = monos.iter().copied().filter(|m| set.contains(m)).collect();
    (cubes, residual)
}

// ANF rewrite of a cube set over its support: expand mixed-polarity cubes
// into positive monomials (canonical, so all cancellation happens), then
// re-express the monomial set as few cubes as found among: an exact minimum
// ESOP (support <= 4), greedy subcube covering + maximum matching on the
// residual, and maximum matching alone (covering can lose when it strands
// monomials the matching wanted). The zero monomial may be consumed by a
// cover/pair cube (as a pure-negative cube) or left to the parity delta.
// Returns (cubes, parity_delta, exact_used) or None when the support or the
// expansion would be too large.
// Canonical ANF of a cube set: the sorted support and the sorted positive
// monomials (bit i = support[i]) that survive cancellation. None when the
// support exceeds `support_cap` (hard cap 63) or the expansion budget.
pub(super) fn anf_expand(cubes: &[Lits], support_cap: usize) -> Option<(Vec<u16>, Vec<u64>)> {
    let mut support: Vec<u16> = cubes
        .iter()
        .flat_map(|c| c.iter().map(|&(w, _)| w))
        .collect();
    support.sort_unstable();
    support.dedup();
    let n = support.len();
    if n > support_cap.min(63) {
        return None;
    }
    let idx_of: FxHashMap<u16, u32> = support
        .iter()
        .enumerate()
        .map(|(i, &w)| (w, i as u32))
        .collect();
    let mut budget = 1u64 << 18;
    let mut anf = FxHashSet::<u64>::default();
    for c in cubes {
        let (mut pos, mut neg) = (0u64, 0u64);
        for &(w, p) in c {
            let b = 1u64 << idx_of[&w];
            if p { pos |= b } else { neg |= b }
        }
        let terms = 1u64 << neg.count_ones();
        if terms > budget {
            return None;
        }
        budget -= terms;
        // cube = AND(pos) * PROD(1 XOR w in neg) = XOR over subsets of neg.
        let mut sub = neg;
        loop {
            let monomial = pos | sub;
            if !anf.insert(monomial) {
                anf.remove(&monomial);
            }
            if sub == 0 {
                break;
            }
            sub = (sub - 1) & neg;
        }
    }
    let mut monos: Vec<u64> = anf.into_iter().collect();
    monos.sort_unstable();
    Some((support, monos))
}

pub(super) fn anf_reduce(cubes: &[Lits], support_cap: usize) -> Option<(Vec<Lits>, bool, bool)> {
    let (support, monos) = anf_expand(cubes, support_cap)?;
    Some(esop_from_monomials(&support, &monos))
}

// Deterministic ESOP from a canonical monomial set over a sorted support
// (bit i of a monomial = support[i], n <= 63): best of the exact minimum
// table (n <= 4), greedy subcube cover + maximum matching, matching alone.
// Depends on nothing but (support, monomials), so it is a function of the
// activation function: applied to a packed ANF gate it yields one
// compacted spelling per function (postprocessing::compress::compact).
// Returns (cubes, parity_delta, exact_used).
pub(super) fn esop_from_monomials(support: &[u16], monos: &[u64]) -> (Vec<Lits>, bool, bool) {
    let n = support.len();
    if monos.is_empty() {
        return (Vec::new(), false, false);
    }
    // Candidate strategies, first-listed wins ties on (cubes, lits).
    let mut cands: Vec<(Vec<(u64, u64)>, bool, bool)> = Vec::new();
    if (1..=4).contains(&n) {
        let (cs, delta) = exact_small(&monos, n);
        cands.push((cs, delta, true));
    }
    {
        let (mut cs, resid) = greedy_cover(&monos, n);
        let (pairs, singles) = match_pairs(&resid, n);
        cs.extend(pairs);
        let mut delta = false;
        for &m in &singles {
            if m == 0 {
                delta = true
            } else {
                cs.push((m, 0))
            }
        }
        cands.push((cs, delta, false));
    }
    {
        let (mut cs, singles) = match_pairs(&monos, n);
        let mut delta = false;
        for &m in &singles {
            if m == 0 {
                delta = true
            } else {
                cs.push((m, 0))
            }
        }
        cands.push((cs, delta, false));
    }
    let (mut best, delta, exact) = cands
        .into_iter()
        .min_by_key(|(cs, _, _)| {
            (
                cs.len(),
                cs.iter()
                    .map(|&(p, ng)| (p | ng).count_ones() as u64)
                    .sum::<u64>(),
            )
        })
        .expect("at least one strategy");
    best.sort_unstable_by_key(|&(p, ng)| (std::cmp::Reverse((p | ng).count_ones()), p, ng));
    let to_lits = |pos: u64, neg: u64| -> Lits {
        let mut l: Lits = Lits::new();
        for (i, &w) in support.iter().enumerate() {
            if pos >> i & 1 == 1 {
                l.push((w, true));
            } else if neg >> i & 1 == 1 {
                l.push((w, false));
            }
        }
        l
    };
    let out: Vec<Lits> = best.iter().map(|&(p, ng)| to_lits(p, ng)).collect();
    (out, delta, exact)
}

// Reduce one gathered group to a minimal-found cube list and emit as XGates
// (parity absorbed as comp on the first cube, or an X gate if none remain).
pub(super) fn reduce_group(
    target: u16,
    members: &[XGate],
    p: &CompressParams,
    rng: &mut StdRng,
    rep: &mut CompressReport,
) -> Vec<XGate> {
    rep.groups += 1;
    rep.max_group = rep.max_group.max(members.len());
    if members.len() == 1 {
        return members.to_vec();
    }
    rep.multi_groups += 1;
    let mut parity = false;
    let mut cubes: Vec<Lits> = Vec::with_capacity(members.len());
    for m in members {
        debug_assert_eq!(m.target, target);
        if m.comp {
            parity = !parity;
        }
        cubes.push(m.ctrls.clone());
    }
    // Pairwise catalogue (cancel / drop-literal / subsume) to a fixed point.
    'outer: loop {
        for i in 0..cubes.len() {
            // `a` depends only on `i`; every merge restarts the scan from the
            // outer loop, so it can never go stale inside the `j` sweep.
            let a = XGate {
                target,
                comp: false,
                ctrls: cubes[i].clone(),
            };
            for j in i + 1..cubes.len() {
                let b = XGate {
                    target,
                    comp: false,
                    ctrls: cubes[j].clone(),
                };
                if let Some(m) = merge_result(&a, &b) {
                    rep.catalogue_merges += 1;
                    let repl = match m {
                        Merge::Cancel => None,
                        Merge::DropLit(g) | Merge::Subsume(g) | Merge::XFuse(g) => Some(g.ctrls),
                        // Both operands are built comp=false two lines above,
                        // and Absorb requires a comp=1 partner, so it cannot
                        // arise here. Assert rather than map, so that changing
                        // the operands fails loudly instead of silently
                        // dropping the comp bit the merge was carrying.
                        Merge::Absorb(_) => {
                            unreachable!("ESOP cubes are comp=0; Absorb needs a comp=1 partner")
                        }
                    };
                    cubes.swap_remove(j);
                    match repl {
                        Some(c) => cubes[i] = c,
                        None => {
                            cubes.swap_remove(i);
                        }
                    }
                    continue 'outer;
                }
            }
        }
        break;
    }
    // ANF alternative: canonical cancellation, kept when strictly smaller.
    // Two-member groups belong here too: pairs whose XOR is one COMPLEMENTED
    // cube (presplit-rejoin, const-XOR-cube) are exactly what the pairwise
    // catalogue must refuse but the parity slot absorbs for free.
    if cubes.len() >= 2 {
        if let Some((alt, delta, exact)) = anf_reduce(&cubes, p.anf_support_cap) {
            let (alt_l, cur_l) = (
                alt.iter().map(|c| c.len()).sum::<usize>(),
                cubes.iter().map(|c| c.len()).sum::<usize>(),
            );
            if (alt.len(), alt_l) < (cubes.len(), cur_l) {
                rep.anf_wins += 1;
                if exact {
                    rep.exact_wins += 1;
                }
                cubes = alt;
                parity ^= delta;
            }
        }
    }
    let mut out: Vec<XGate> = Vec::with_capacity(cubes.len().max(1));
    for (k, c) in cubes.into_iter().enumerate() {
        out.push(XGate {
            target,
            comp: parity && k == 0,
            ctrls: c,
        });
    }
    if out.is_empty() && parity {
        out.push(XGate::x_gate(target));
    }
    if p.local_verify {
        // An identity reduction (output == input gate-for-gate) is functionally
        // equal by construction; verifying it exhaustively is pure waste and
        // ~95% of multi-group reductions are identities. The verify rng feeds
        // nothing but the assertion, so skipping its draws cannot reach the
        // output bytes.
        if out == members {
            rep.verifies_skipped += 1;
        } else {
            verify_group(members, &out, rng);
        }
    }
    out
}

// Per-gate bitmask compilation over the indexed sorted support: each cube is
// flattened at `words` u64s per gate as (positive-literal, negative-literal)
// masks plus its comp bit.  Bit `i` of an assignment corresponds to
// `support[i]`.
pub(super) fn cube_masks(
    gates: &[XGate],
    support: &[u16],
    words: usize,
) -> (Vec<bool>, Vec<u64>, Vec<u64>) {
    let mut comps = Vec::with_capacity(gates.len());
    let mut pos = vec![0u64; gates.len() * words];
    let mut neg = vec![0u64; gates.len() * words];
    for (i, g) in gates.iter().enumerate() {
        comps.push(g.comp);
        for &(w, p) in &g.ctrls {
            let bit = support
                .binary_search(&w)
                .expect("verify support must contain every cube wire");
            let slot = i * words + bit / 64;
            let mask = 1u64 << (bit % 64);
            if p {
                pos[slot] |= mask;
            } else {
                neg[slot] |= mask;
            }
        }
    }
    (comps, pos, neg)
}

// XOR of the compiled cubes on one assignment bitset: a cube fires iff every
// positive bit is set and every negative bit is clear.
pub(super) fn masked_parity(
    comps: &[bool],
    pos: &[u64],
    neg: &[u64],
    words: usize,
    assign: &[u64],
) -> bool {
    let mut acc = false;
    for (i, &comp) in comps.iter().enumerate() {
        let base = i * words;
        let mut fires = true;
        for word in 0..words {
            let a = assign[word];
            if a & pos[base + word] != pos[base + word] || a & neg[base + word] != 0 {
                fires = false;
                break;
            }
        }
        acc ^= comp ^ fires;
    }
    acc
}

/// Original dyn-Fn cube evaluation, retained as the equivalence reference for
/// `opt_equiv_masked_parity_matches_reference`.
#[cfg(test)]
pub(super) fn parity_of_reference(gates: &[XGate], val: &dyn Fn(u16) -> bool) -> bool {
    let mut acc = false;
    for g in gates {
        acc ^= g.comp ^ g.ctrls.iter().all(|&(w, p)| val(w) == p);
    }
    acc
}

pub(super) fn verify_group(before: &[XGate], after: &[XGate], rng: &mut StdRng) {
    let mut support: Vec<u16> = before
        .iter()
        .chain(after)
        .flat_map(|g| g.ctrls.iter().map(|&(w, _)| w))
        .collect();
    support.sort_unstable();
    support.dedup();
    let words = support.len().div_ceil(64).max(1);
    let (before_comps, before_pos, before_neg) = cube_masks(before, &support, words);
    let (after_comps, after_pos, after_neg) = cube_masks(after, &support, words);
    let check = |assign: &[u64]| {
        assert_eq!(
            masked_parity(&before_comps, &before_pos, &before_neg, words, assign),
            masked_parity(&after_comps, &after_pos, &after_neg, words, assign),
            "fcompress group reduction changed the function"
        );
    };
    if support.len() <= 16 {
        for a in 0u32..(1u32 << support.len()) {
            check(&[a as u64]);
        }
    } else {
        let mut assign = vec![0u64; words];
        for _ in 0..512 {
            assign.fill(0);
            // One draw per support wire in ascending wire order: the exact
            // sequence the HashMap-based sampler drew.
            for bit in 0..support.len() {
                if rng.random_bool(0.5) {
                    assign[bit / 64] |= 1u64 << (bit % 64);
                }
            }
            check(&assign);
        }
    }
}
