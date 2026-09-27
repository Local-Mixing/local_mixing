//! mint_long_identities — MANUFACTURE locally-geodesic g57 identities of
//! length 13–17 by pairing stored minimal circuits with MINTED longer
//! realizations, using the frozen store as a synthesis oracle.
//!
//! The store maps a canonical function key to the chain of ALL its <=6-gate
//! circuits (complete g57 ball of diameter 6, plus partial shells 7–9 with one
//! circuit each). Per-key chains therefore top out at 6 gates, so same-key
//! pairing (extract_identities) cannot reach identities longer than ~12–13.
//! This tool MINTS the missing longer realizations:
//!
//! For a key f with a stored minimal circuit A (|A| = kmin), enumerate suffix
//! words s of length r (radius) over A's wire frame plus r fresh wires. The
//! word P = A ++ reverse(s) computes perm(s)^-1 ∘ f (g57 gates are
//! involutions). Canonicalize P (forward AND reversed key, exactly like the
//! runtime lookup) and get_regular; every stored circuit B' of that key,
//! relabeled back to A's frame, gives B = B' ++ s, a (|B'|+r)-gate realization
//! of f. Then C = A ++ reverse(B) is an identity of length |A|+|B'|+r.
//! Keep C only if it is reduced, contains no proper contiguous sub-identity
//! (SMID / locally geodesic) and re-verifies as an exact identity; canonicalize
//! survivors with dihedral_canonical_word and dedup globally.
//!
//! Tiers (kmin = entry's minimal stored length):
//!   kmin=6, r=1 : 6+7 = 13 (bulk), minted-pair 7+7 = 14
//!   kmin=7, r=1 : 7+7 = 14; 7+8 = 15 (8 = 7-gate shell member + 1 suffix gate)
//!   kmin=7, r=2 : more 15s on a small subset (6-gate member + 2 suffix gates)
//!   kmin=8, r<=2: 8+8 = 16 (and 8+9 = 17 when targeted)
//!
//! Usage:
//!   mint_long_identities <store_dir>
//!     [--keys6 N] [--keys7 N] [--keys7-r2 N] [--keys8 N]     (PER SHARD quotas)
//!     [--target-lens 13,14,15,16] [--out FILE] [--shards N]
//!     [--suffix-cap N] [--minted-cap N] [--mins-cap N] [--max-support W]
//!     [--native|--swap|--auto]
//!
//! Every random choice (suffix order, radius-2 subsampling) is seeded from OS
//! entropy.

use local_mixing::circuit::{CircuitSeq, Gate as G57Gate, Permutation};
use local_mixing::db_generation::curated_full::{dihedral_canonical_word, word_bytes};
use local_mixing::db_mixing::frozen::{scan_shard, FrozenDb};
use rand::rngs::StdRng;
use rand::seq::SliceRandom;
use rand::{Rng, SeedableRng};
use rayon::prelude::*;
use std::collections::{BTreeMap, HashSet};
use std::io::Write;
use std::sync::atomic::{AtomicU64, Ordering};
use std::time::Instant;

type Gate = [u16; 3];

// ---------------------------------------------------------------------------
// Store-value parsing + identity predicates (mirrors extract_identities.rs)
// ---------------------------------------------------------------------------

/// Split a legacy value into its circuits; swap the two controls per gate when
/// `swap` (the store's control convention).
fn parse_value(value: &[u8], swap: bool) -> Vec<Vec<Gate>> {
    let mut out = Vec::new();
    let mut pos = 0usize;
    while pos < value.len() {
        let len = value[pos] as usize;
        pos += 1;
        if len == 0 || len % 3 != 0 || pos + len > value.len() {
            break;
        }
        let mut g = Vec::with_capacity(len / 3);
        for ch in value[pos..pos + len].chunks_exact(3) {
            if swap {
                g.push([ch[0] as u16, ch[2] as u16, ch[1] as u16]);
            } else {
                g.push([ch[0] as u16, ch[1] as u16, ch[2] as u16]);
            }
        }
        out.push(g);
        pos += len;
    }
    out
}

#[inline]
fn width(word: &[Gate]) -> u32 {
    word.iter()
        .flat_map(|g| g.iter().copied())
        .max()
        .map_or(0, |m| m as u32 + 1)
}

/// Width-independent probe identity test (never falsely accepts in practice);
/// survivors are re-checked exactly by `is_identity_exact` before output.
const PROBES: usize = 96;
fn is_identity(word: &[Gate]) -> bool {
    let w = width(word);
    if w == 0 {
        return true;
    }
    let mask = if w >= 63 { u64::MAX } else { (1u64 << w) - 1 };
    let mut x = 0x9e3779b97f4a7c15u64;
    for _ in 0..PROBES {
        x ^= x << 13;
        x ^= x >> 7;
        x ^= x << 17;
        let s = x & mask;
        if G57Gate::evaluate_index_list_64(s, word) != s {
            return false;
        }
    }
    true
}

/// Exact identity test (all 2^w states) for small support; wide probe fallback.
fn is_identity_exact(word: &[Gate]) -> bool {
    let w = width(word);
    if w == 0 {
        return true;
    }
    if w <= 22 {
        for s in 0u64..(1u64 << w) {
            if G57Gate::evaluate_index_list_64(s, word) != s {
                return false;
            }
        }
        true
    } else {
        let mask = if w >= 63 { u64::MAX } else { (1u64 << w) - 1 };
        let mut x = 0x1234567deadbeefu64;
        for _ in 0..4096 {
            x ^= x << 13;
            x ^= x >> 7;
            x ^= x << 17;
            if G57Gate::evaluate_index_list_64(x & mask, word) != x & mask {
                return false;
            }
        }
        true
    }
}

#[inline]
fn commute(g: &Gate, h: &Gate) -> bool {
    g[0] != h[0] && g[0] != h[1] && g[0] != h[2] && h[0] != g[1] && h[0] != g[2]
}

/// No adjacent-equal gate and no equal pair separated only by commuting gates.
fn reduced(w: &[Gate]) -> bool {
    let n = w.len();
    for i in 0..n {
        for j in (i + 1)..n {
            if w[i] == w[j] && (i + 1..j).all(|k| commute(&w[i], &w[k])) {
                return false;
            }
        }
    }
    true
}

/// Locally geodesic (SMID): reduced, and no proper contiguous subword is
/// itself an identity.
fn locally_geodesic(w: &[Gate]) -> bool {
    let n = w.len();
    if !reduced(w) {
        return false;
    }
    for i in 0..n {
        for j in (i + 2)..=n {
            if j - i < n && is_identity(&w[i..j]) {
                return false;
            }
        }
    }
    true
}

/// Lexicographically-least word of a commutation (trace) class: greedily pick
/// the smallest gate that can be moved to the front (commutes with everything
/// before it). Two words are commutation-equivalent iff their normal forms are
/// equal. Used to drop MINTED realizations that are mere commuting reorders of
/// a stored minimal circuit (those never survive `reduced` after pairing).
fn trace_nf(w: &[Gate]) -> Vec<Gate> {
    let mut rem: Vec<Gate> = w.to_vec();
    let mut out = Vec::with_capacity(rem.len());
    while !rem.is_empty() {
        let mut best: Option<usize> = None;
        for i in 0..rem.len() {
            if rem[..i].iter().all(|h| commute(&rem[i], h)) {
                if best.is_none_or(|b| rem[i] < rem[b]) {
                    best = Some(i);
                }
            }
        }
        out.push(rem.remove(best.unwrap()));
    }
    out
}

fn identity_from(a: &[Gate], b: &[Gate]) -> Vec<Gate> {
    let mut v = Vec::with_capacity(a.len() + b.len());
    v.extend_from_slice(a);
    v.extend(b.iter().rev().copied());
    v
}

fn bytes_to_gates(wb: &[u8]) -> Vec<Gate> {
    wb.chunks_exact(6)
        .map(|c| {
            [
                u16::from_le_bytes([c[0], c[1]]),
                u16::from_le_bytes([c[2], c[3]]),
                u16::from_le_bytes([c[4], c[5]]),
            ]
        })
        .collect()
}

fn shard_list(dir: &str, limit: Option<usize>) -> Vec<usize> {
    let mut v: Vec<usize> = std::fs::read_dir(dir)
        .expect("read store dir")
        .filter_map(|e| {
            let name = e.ok()?.file_name().into_string().ok()?;
            usize::from_str_radix(name.strip_prefix("shard_")?.strip_suffix(".frz")?, 16).ok()
        })
        .collect();
    v.sort_unstable();
    if let Some(n) = limit {
        v.truncate(n);
    }
    v
}

/// Detect the control convention: sample values on shard 0, count same-key
/// pairs that verify as identities under each convention, return the winner.
fn detect_convention(dir: &str, max_len: usize) -> bool {
    let mut native = 0u64;
    let mut swapped = 0u64;
    let mut seen = 0u64;
    scan_shard(dir, 0, &mut |value: &[u8]| {
        if seen >= 4000 {
            return;
        }
        for &swap in &[false, true] {
            let cs = parse_value(value, swap);
            let m = cs.len();
            for i in 0..m {
                for j in (i + 1)..m {
                    if cs[i].len() + cs[j].len() > max_len {
                        continue;
                    }
                    let ident = identity_from(&cs[i], &cs[j]);
                    if is_identity(&ident) {
                        if swap {
                            swapped += 1
                        } else {
                            native += 1
                        }
                    }
                    seen += 1;
                }
            }
        }
    });
    eprintln!("[convention] verified pairs — native {native}, swapped {swapped}");
    swapped > native
}

// ---------------------------------------------------------------------------
// Exact word-equality on small frames (mirrors db_unit_synth.rs)
// ---------------------------------------------------------------------------

#[inline]
fn apply_gate_state(g: Gate, s: usize) -> usize {
    let (a, x, y) = (g[0] as usize, g[1] as usize, g[2] as usize);
    let fire = ((s >> x) & 1) | (1 - ((s >> y) & 1));
    s ^ (fire << a)
}

/// All g57 gates on m compact wires with three DISTINCT wires.
fn gates_on(m: usize) -> Vec<Gate> {
    let mut v = Vec::new();
    for a in 0..m {
        for x in 0..m {
            for y in 0..m {
                if a != x && a != y && x != y {
                    v.push([a as u16, x as u16, y as u16]);
                }
            }
        }
    }
    v
}

fn max_wire_w(word: &[Gate]) -> usize {
    word.iter().flat_map(|g| g.iter().copied()).max().unwrap_or(0) as usize
}

/// Exact permutation table over 2^w states for a word on wires 0..w-1.
fn word_table(word: &[Gate], w: usize) -> Vec<u32> {
    let n = 1usize << w;
    let mut t: Vec<u32> = (0..n as u32).collect();
    for &g in word {
        for s in 0..n {
            t[s] = apply_gate_state(g, t[s] as usize) as u32;
        }
    }
    t
}

/// True iff words `a` and `b` compute the same permutation. Exact up to 16
/// wires; 256-probe fallback above (final output is exact-rechecked anyway).
fn words_equal(a: &[Gate], b: &[Gate]) -> bool {
    let w = max_wire_w(a).max(max_wire_w(b)) + 1;
    if w > 16 {
        let mask: u64 = if w >= 64 { u64::MAX } else { (1u64 << w) - 1 };
        let mut x = 0x9e3779b97f4a7c15u64;
        for _ in 0..256 {
            x ^= x << 13;
            x ^= x >> 7;
            x ^= x << 17;
            let s = x & mask;
            if G57Gate::evaluate_index_list_64(s, a) != G57Gate::evaluate_index_list_64(s, b) {
                return false;
            }
        }
        return true;
    }
    word_table(a, w) == word_table(b, w)
}

// ---------------------------------------------------------------------------
// Lookup + relabel-back (mirrors db_unit_synth.rs / replace.rs)
// ---------------------------------------------------------------------------

/// Canonicalize P (forward then reversed) and get_regular. On the first hit
/// return (value, order, used, is_reversed).
fn lookup(db: &FrozenDb, p: &CircuitSeq) -> Option<(Vec<u8>, Permutation, Vec<u16>, bool)> {
    let (fk, fo, used) = p.canonicalize_polys_single_hashed(false);
    if let Some(fk) = fk {
        if let Some(v) = db.get_regular(&fk) {
            return Some((v, fo, used, false));
        }
    }
    let (rk, ro, used2) = p.canonicalize_polys_single_hashed(true);
    if let Some(rk) = rk {
        if let Some(v) = db.get_regular(&rk) {
            return Some((v, ro, used2, true));
        }
    }
    None
}

/// Relabel a decoded DB candidate (canonical wire space, forward orientation)
/// back to the query's wire frame. Mirrors db_unit_synth.rs exactly.
fn rewire_candidate(
    mut repl: CircuitSeq,
    is_reversed: bool,
    final_order: &Permutation,
    used: &[u16],
    n: usize,
    rng: &mut StdRng,
) -> Vec<Gate> {
    if is_reversed {
        repl.gates.reverse();
    }
    let repl_n = repl.max_wire() as usize + 1;
    let mut order_data = final_order.data.clone();
    while order_data.len() < repl_n {
        let i = order_data.len();
        order_data.push(i);
    }
    repl.rewire(
        &Permutation {
            data: order_data.clone(),
        },
        std::cmp::max(repl_n, final_order.data.len()),
    );
    let repl_n_b = repl.max_wire() as usize + 1;
    let mut used_ext = used.to_vec();
    if used_ext.len() < repl_n_b {
        let mut used_mask = vec![false; n];
        for &w in used_ext.iter() {
            if (w as usize) < n {
                used_mask[w as usize] = true;
            }
        }
        let mut available: Vec<u16> = (0..n as u16).filter(|&w| !used_mask[w as usize]).collect();
        available.shuffle(rng);
        let mut avail = available.into_iter();
        while used_ext.len() < repl_n_b {
            match avail.next() {
                Some(w) => used_ext.push(w),
                None => used_ext.push((n + used_ext.len()) as u16),
            }
        }
    }
    CircuitSeq::unrewire_subcircuit(&repl, &used_ext).gates
}

/// Decode a hit value chain into member words relabeled to the query frame.
fn decode_members(
    value: &[u8],
    swap: bool,
    is_rev: bool,
    order: &Permutation,
    used: &[u16],
    frame_n: usize,
    seed: u64,
) -> Vec<Vec<Gate>> {
    let mut out = Vec::new();
    let mut rng = StdRng::seed_from_u64(seed);
    for gates in parse_value(value, swap) {
        let cand = CircuitSeq { gates };
        out.push(rewire_candidate(cand, is_rev, order, used, frame_n, &mut rng));
    }
    out
}

// ---------------------------------------------------------------------------
// Sampling + minting
// ---------------------------------------------------------------------------

struct Entry {
    kmin: usize,
    mins: Vec<Vec<Gate>>, // stored minimal circuits (common canonical frame)
}

/// Per-shard sampled entries by tier index (0: kmin=6, 1: kmin=7, 2: kmin=8).
fn sample_entries(
    dir: &str,
    shards: &[usize],
    swap: bool,
    quotas: [usize; 3],
    mins_keep: usize,
) -> Vec<[Vec<Entry>; 3]> {
    shards
        .par_iter()
        .map(|&s| {
            let mut got: [Vec<Entry>; 3] = [Vec::new(), Vec::new(), Vec::new()];
            scan_shard(dir, s, &mut |value: &[u8]| {
                if got[0].len() >= quotas[0]
                    && got[1].len() >= quotas[1]
                    && got[2].len() >= quotas[2]
                {
                    return;
                }
                let cs = parse_value(value, swap);
                if cs.is_empty() {
                    return;
                }
                let kmin = cs.iter().map(|c| c.len()).min().unwrap();
                let ti = match kmin {
                    6 => 0usize,
                    7 => 1,
                    8 => 2,
                    _ => return,
                };
                if got[ti].len() >= quotas[ti] {
                    return;
                }
                let mins: Vec<Vec<Gate>> = cs
                    .into_iter()
                    .filter(|c| c.len() == kmin)
                    .take(mins_keep)
                    .collect();
                got[ti].push(Entry { kmin, mins });
            });
            got
        })
        .collect()
}

#[derive(Default)]
struct TierStats {
    done: AtomicU64,
    skipped_support: AtomicU64,
    lookups: AtomicU64,
    hits: AtomicU64,
    minted_total: AtomicU64,
    keys_with_mint: AtomicU64,
    verify_fail: AtomicU64,
    trivial_reorders: AtomicU64,
    idents: AtomicU64,
    cand: AtomicU64,
    f_ident: AtomicU64,
    f_reduced: AtomicU64,
    f_smid: AtomicU64,
    dumped: AtomicU64,
}

struct TierCfg {
    label: &'static str,
    radius: usize,
    targets: HashSet<usize>,
    suffix_cap: usize,
    minted_cap: usize,
    mins_cap: usize,
    max_support: usize,
    total_entries: usize,
}

fn process_entry(
    db: &FrozenDb,
    e: &Entry,
    cfg: &TierCfg,
    swap: bool,
    seed: u64,
    st: &TierStats,
) -> HashSet<Vec<u8>> {
    let mut out: HashSet<Vec<u8>> = HashSet::new();
    let kmin = e.kmin;
    let t: &Vec<Gate> = &e.mins[0];
    let w = e.mins.iter().map(|c| width(c)).max().unwrap_or(0) as usize;
    let done = st.done.fetch_add(1, Ordering::Relaxed) + 1;
    if done % 8192 == 0 {
        eprintln!(
            "[{}] {done}/{} keys  lookups={} hits={} minted={} keys_with_mint={} idents={}",
            cfg.label,
            cfg.total_entries,
            st.lookups.load(Ordering::Relaxed),
            st.hits.load(Ordering::Relaxed),
            st.minted_total.load(Ordering::Relaxed),
            st.keys_with_mint.load(Ordering::Relaxed),
            st.idents.load(Ordering::Relaxed),
        );
    }
    if w < 3 || w > cfg.max_support {
        st.skipped_support.fetch_add(1, Ordering::Relaxed);
        return out;
    }
    let mut rng = StdRng::seed_from_u64(seed);
    let accept_len = |l: usize| -> bool {
        l >= kmin && (cfg.targets.contains(&(kmin + l)) || cfg.targets.contains(&(2 * l)))
    };
    let frame_n = w + cfg.radius; // fresh suffix wires at w..w+radius-1
    let mut minted: Vec<Vec<Gate>> = Vec::new();
    let mut seen: HashSet<Vec<u8>> = HashSet::new();
    // trace normal forms of the stored minimal circuits: a minted realization
    // of length kmin whose normal form matches is a commuting reorder of a
    // stored circuit — never a genuine second realization.
    let min_nfs: HashSet<Vec<u8>> = e
        .mins
        .iter()
        .map(|c| word_bytes(&trace_nf(c)))
        .collect();

    // Collect verified minted realizations from one lookup hit.
    let absorb = |value: &[u8],
                      ord: &Permutation,
                      used: &[u16],
                      is_rev: bool,
                      s_word: &[Gate],
                      minted: &mut Vec<Vec<Gate>>,
                      seen: &mut HashSet<Vec<u8>>,
                      dseed: u64|
     -> bool {
        for m in decode_members(value, swap, is_rev, ord, used, frame_n, dseed) {
            let mut b = m;
            b.extend_from_slice(s_word);
            if !accept_len(b.len()) {
                continue;
            }
            if !words_equal(&b, t) {
                st.verify_fail.fetch_add(1, Ordering::Relaxed);
                continue;
            }
            if b.len() == kmin && min_nfs.contains(&word_bytes(&trace_nf(&b))) {
                st.trivial_reorders.fetch_add(1, Ordering::Relaxed);
                continue;
            }
            if seen.insert(word_bytes(&b)) {
                minted.push(b);
                if minted.len() >= cfg.minted_cap {
                    return true; // cap reached
                }
            }
        }
        false
    };

    // --- radius 1: suffix = one gate over the frame plus one fresh wire ---
    let mut g1s = gates_on(w + 1);
    g1s.shuffle(&mut rng);
    for (gi, &g) in g1s.iter().enumerate() {
        if minted.len() >= cfg.minted_cap {
            break;
        }
        let mut pg = t.clone();
        pg.push(g); // reverse([g]) == [g]
        let p = CircuitSeq { gates: pg };
        st.lookups.fetch_add(1, Ordering::Relaxed);
        if let Some((value, ord, used, is_rev)) = lookup(db, &p) {
            st.hits.fetch_add(1, Ordering::Relaxed);
            if absorb(
                &value,
                &ord,
                &used,
                is_rev,
                &[g],
                &mut minted,
                &mut seen,
                seed ^ ((gi as u64) << 8) ^ 0x5eed_0001,
            ) {
                break;
            }
        }
    }

    // --- radius 2: suffix = two gates over the frame plus two fresh wires ---
    if cfg.radius >= 2 && minted.len() < cfg.minted_cap {
        let g2s = gates_on(w + 2);
        let total = (g2s.len() as f64) * (g2s.len() as f64);
        let p_acc = (cfg.suffix_cap as f64 / total).min(1.0);
        let mut order: Vec<usize> = (0..g2s.len()).collect();
        order.shuffle(&mut rng);
        'outer2: for &i1 in &order {
            if minted.len() >= cfg.minted_cap {
                break;
            }
            let g1 = g2s[i1];
            for (i2, &g2) in g2s.iter().enumerate() {
                if g1 == g2 {
                    continue; // cancels
                }
                if commute(&g1, &g2) && g1 > g2 {
                    continue; // dedup commuting orders
                }
                if p_acc < 1.0 && rng.random::<f64>() >= p_acc {
                    continue;
                }
                // s = [g1, g2]; P = T ++ reverse(s) = T ++ [g2, g1]
                let mut pg = t.clone();
                pg.push(g2);
                pg.push(g1);
                let p = CircuitSeq { gates: pg };
                st.lookups.fetch_add(1, Ordering::Relaxed);
                if let Some((value, ord, used, is_rev)) = lookup(db, &p) {
                    st.hits.fetch_add(1, Ordering::Relaxed);
                    if absorb(
                        &value,
                        &ord,
                        &used,
                        is_rev,
                        &[g1, g2],
                        &mut minted,
                        &mut seen,
                        seed ^ ((i1 as u64) << 24) ^ ((i2 as u64) << 8) ^ 0x5eed_0002,
                    ) {
                        break 'outer2;
                    }
                }
            }
        }
    }

    if minted.is_empty() {
        return out;
    }
    st.keys_with_mint.fetch_add(1, Ordering::Relaxed);
    st.minted_total
        .fetch_add(minted.len() as u64, Ordering::Relaxed);

    // --- assemble identities: stored-min x minted, and minted x minted ---
    let mut consider = |ident: Vec<Gate>| {
        if !cfg.targets.contains(&ident.len()) {
            return;
        }
        st.cand.fetch_add(1, Ordering::Relaxed);
        if !is_identity(&ident) {
            st.f_ident.fetch_add(1, Ordering::Relaxed);
            return;
        }
        if !reduced(&ident) {
            st.f_reduced.fetch_add(1, Ordering::Relaxed);
            if st.dumped.fetch_add(1, Ordering::Relaxed) < 3 {
                eprintln!("[{}][dbg] REDUCED-fail: {:?}", cfg.label, ident);
            }
            return;
        }
        if !locally_geodesic(&ident) {
            st.f_smid.fetch_add(1, Ordering::Relaxed);
            // find and report the first offending sub-identity
            if st.dumped.fetch_add(1, Ordering::Relaxed) < 3 {
                let n = ident.len();
                'find: for i in 0..n {
                    for j in (i + 2)..=n {
                        if j - i < n && is_identity(&ident[i..j]) {
                            eprintln!(
                                "[{}][dbg] SMID-fail sub [{i}..{j}) of {n}: {:?}",
                                cfg.label, ident
                            );
                            break 'find;
                        }
                    }
                }
            }
            return;
        }
        out.insert(word_bytes(&dihedral_canonical_word(&ident)));
    };
    for a in e.mins.iter().take(cfg.mins_cap) {
        for b in &minted {
            if cfg.targets.contains(&(a.len() + b.len())) {
                consider(identity_from(a, b));
            }
        }
    }
    for i in 0..minted.len() {
        for j in (i + 1)..minted.len() {
            if cfg.targets.contains(&(minted[i].len() + minted[j].len())) {
                consider(identity_from(&minted[i], &minted[j]));
            }
        }
    }
    st.idents.fetch_add(out.len() as u64, Ordering::Relaxed);
    out
}

// ---------------------------------------------------------------------------
// CLI + main
// ---------------------------------------------------------------------------

struct Args {
    dir: String,
    keys6: usize,
    keys7: usize,
    keys7_r2: usize,
    keys8: usize,
    target_lens: HashSet<usize>,
    out: Option<String>,
    shards: Option<usize>,
    suffix_cap: usize,
    minted_cap: usize,
    mins_cap: usize,
    max_support: usize,
    max_support8: usize,
    conv: Option<bool>,
}

fn parse_args() -> Args {
    let mut a = std::env::args().skip(1);
    let dir = a
        .next()
        .expect("usage: mint_long_identities <store_dir> [opts]");
    let mut args = Args {
        dir,
        keys6: 1200,
        keys7: 800,
        keys7_r2: 12,
        keys8: 20,
        target_lens: [13usize, 14, 15, 16].into_iter().collect(),
        out: None,
        shards: None,
        suffix_cap: 30_000,
        minted_cap: 12,
        mins_cap: 8,
        max_support: 14,
        max_support8: 20,
        conv: None,
    };
    while let Some(tok) = a.next() {
        match tok.as_str() {
            "--keys6" => args.keys6 = a.next().unwrap().parse().unwrap(),
            "--keys7" => args.keys7 = a.next().unwrap().parse().unwrap(),
            "--keys7-r2" => args.keys7_r2 = a.next().unwrap().parse().unwrap(),
            "--keys8" => args.keys8 = a.next().unwrap().parse().unwrap(),
            "--target-lens" => {
                args.target_lens = a
                    .next()
                    .unwrap()
                    .split(',')
                    .map(|s| s.trim().parse().unwrap())
                    .collect()
            }
            "--out" => args.out = Some(a.next().unwrap()),
            "--shards" => args.shards = Some(a.next().unwrap().parse().unwrap()),
            "--suffix-cap" => args.suffix_cap = a.next().unwrap().parse().unwrap(),
            "--minted-cap" => args.minted_cap = a.next().unwrap().parse().unwrap(),
            "--mins-cap" => args.mins_cap = a.next().unwrap().parse().unwrap(),
            "--max-support" => args.max_support = a.next().unwrap().parse().unwrap(),
            "--max-support8" => args.max_support8 = a.next().unwrap().parse().unwrap(),
            "--native" => args.conv = Some(false),
            "--swap" => args.conv = Some(true),
            "--auto" => args.conv = None,
            other => panic!("unknown arg {other}"),
        }
    }
    args
}

fn write_csv(path: &str, set: &HashSet<Vec<u8>>) {
    let mut by_len: BTreeMap<usize, Vec<&Vec<u8>>> = BTreeMap::new();
    for w in set {
        by_len.entry(w.len() / 6).or_default().push(w);
    }
    let mut f = std::io::BufWriter::new(std::fs::File::create(path).unwrap());
    writeln!(f, "length,n_wires,circuit").unwrap();
    for (len, v) in &mut by_len {
        v.sort();
        for wb in v.iter() {
            let gates = bytes_to_gates(wb);
            let nw = width(&gates);
            let s: Vec<String> = gates
                .iter()
                .map(|g| format!("{},{},{}", g[0], g[1], g[2]))
                .collect();
            writeln!(f, "{len},{nw},{}", s.join(";")).unwrap();
        }
    }
}

fn main() {
    let args = parse_args();
    let swap = args
        .conv
        .unwrap_or_else(|| detect_convention(&args.dir, 12));
    eprintln!(
        "[mint] store={} convention={} targets={:?}",
        args.dir,
        if swap { "swapped-controls" } else { "native" },
        {
            let mut t: Vec<_> = args.target_lens.iter().collect();
            t.sort();
            t
        }
    );
    let shards = shard_list(&args.dir, args.shards);
    let nsh = shards.len();
    eprintln!("[mint] scanning {nsh} shards for kmin=6/7/8 entries");
    let db = FrozenDb::open(&args.dir, None);

    let t_scan = Instant::now();
    let per_shard = sample_entries(
        &args.dir,
        &shards,
        swap,
        [args.keys6, args.keys7 + args.keys7_r2, args.keys8],
        24,
    );
    // Merge; tier7 split per shard into a radius-2 head and radius-1 tail.
    let mut tier6: Vec<Entry> = Vec::new();
    let mut tier7_r1: Vec<Entry> = Vec::new();
    let mut tier7_r2: Vec<Entry> = Vec::new();
    let mut tier8: Vec<Entry> = Vec::new();
    for [t6, t7, t8] in per_shard {
        tier6.extend(t6);
        let mut it = t7.into_iter();
        for _ in 0..args.keys7_r2 {
            match it.next() {
                Some(e) => tier7_r2.push(e),
                None => break,
            }
        }
        tier7_r1.extend(it);
        tier8.extend(t8);
    }
    eprintln!(
        "[mint] sampled in {:.1}s: kmin6={} kmin7(r1)={} kmin7(r2)={} kmin8={}",
        t_scan.elapsed().as_secs_f64(),
        tier6.len(),
        tier7_r1.len(),
        tier7_r2.len(),
        tier8.len()
    );
    // support histograms of minimal circuits (widths drive suffix cost)
    for (label, tier) in [
        ("tier6", &tier6),
        ("tier7", &tier7_r1),
        ("tier8", &tier8),
    ] {
        let mut hist: BTreeMap<u32, usize> = BTreeMap::new();
        for e in tier.iter().take(20000) {
            *hist.entry(width(&e.mins[0])).or_default() += 1;
        }
        eprintln!("[mint] {label} support histogram (first 20k): {hist:?}");
    }

    let master: u64 = rand::random();
    let mut global: HashSet<Vec<u8>> = HashSet::new();

    let tiers: Vec<(Vec<Entry>, TierCfg)> = vec![
        (
            tier6,
            TierCfg {
                label: "kmin6-r1",
                radius: 1,
                targets: args.target_lens.clone(),
                suffix_cap: args.suffix_cap,
                minted_cap: args.minted_cap,
                mins_cap: args.mins_cap,
                max_support: args.max_support,
                total_entries: 0,
            },
        ),
        (
            tier7_r1,
            TierCfg {
                label: "kmin7-r1",
                radius: 1,
                targets: args.target_lens.clone(),
                suffix_cap: args.suffix_cap,
                minted_cap: args.minted_cap,
                mins_cap: args.mins_cap,
                max_support: args.max_support,
                total_entries: 0,
            },
        ),
        (
            tier7_r2,
            TierCfg {
                label: "kmin7-r2",
                radius: 2,
                targets: args.target_lens.clone(),
                suffix_cap: args.suffix_cap,
                minted_cap: args.minted_cap,
                mins_cap: args.mins_cap,
                max_support: args.max_support,
                total_entries: 0,
            },
        ),
        (
            tier8,
            TierCfg {
                label: "kmin8-r2",
                radius: 2,
                targets: args.target_lens.clone(),
                suffix_cap: args.suffix_cap,
                minted_cap: args.minted_cap,
                mins_cap: args.mins_cap,
                max_support: args.max_support8,
                total_entries: 0,
            },
        ),
    ];

    for (entries, mut cfg) in tiers {
        if entries.is_empty() {
            continue;
        }
        cfg.total_entries = entries.len();
        let st = TierStats::default();
        let t0 = Instant::now();
        let sets: Vec<HashSet<Vec<u8>>> = entries
            .par_iter()
            .enumerate()
            .map(|(i, e)| {
                let seed = master ^ (i as u64).wrapping_mul(0x9E37_79B9_7F4A_7C15)
                    ^ ((cfg.radius as u64) << 60);
                process_entry(&db, e, &cfg, swap, seed, &st)
            })
            .collect();
        let mut tier_set: HashSet<Vec<u8>> = HashSet::new();
        for s in sets {
            tier_set.extend(s);
        }
        let hit_keys = st.keys_with_mint.load(Ordering::Relaxed);
        eprintln!(
            "[{}] DONE {:.1}s: keys={} skipped_support={} lookups={} hits={} \
             minted={} keys_with_mint={} ({:.1}%) trivial_reorders={} verify_fail={} \
             idents(raw)={} distinct={}",
            cfg.label,
            t0.elapsed().as_secs_f64(),
            entries.len(),
            st.skipped_support.load(Ordering::Relaxed),
            st.lookups.load(Ordering::Relaxed),
            st.hits.load(Ordering::Relaxed),
            st.minted_total.load(Ordering::Relaxed),
            hit_keys,
            100.0 * hit_keys as f64 / entries.len() as f64,
            st.trivial_reorders.load(Ordering::Relaxed),
            st.verify_fail.load(Ordering::Relaxed),
            st.idents.load(Ordering::Relaxed),
            tier_set.len()
        );
        eprintln!(
            "[{}] candidates={} fail_ident={} fail_reduced={} fail_smid={}",
            cfg.label,
            st.cand.load(Ordering::Relaxed),
            st.f_ident.load(Ordering::Relaxed),
            st.f_reduced.load(Ordering::Relaxed),
            st.f_smid.load(Ordering::Relaxed)
        );
        global.extend(tier_set);
        if let Some(path) = &args.out {
            write_csv(path, &global);
            eprintln!("[mint] snapshot: {} identities -> {path}", global.len());
        }
    }

    // Exact re-verification of the (probe-accepted) survivor set.
    let before = global.len();
    let verified: HashSet<Vec<u8>> = global
        .into_par_iter()
        .filter(|wb| is_identity_exact(&bytes_to_gates(wb)))
        .collect();
    if verified.len() != before {
        eprintln!(
            "[verify] {} of {before} survivors failed exact re-check (dropped)",
            before - verified.len()
        );
    } else {
        eprintln!("[verify] all {before} survivors passed exact re-check");
    }

    let mut by_len: BTreeMap<usize, usize> = BTreeMap::new();
    for w in &verified {
        *by_len.entry(w.len() / 6).or_default() += 1;
    }
    eprintln!("\n== distinct locally-geodesic identities by length ==");
    for (len, c) in &by_len {
        eprintln!("  {len:>3} gates : {c}");
    }
    eprintln!("  TOTAL     : {}", verified.len());
    if let Some(path) = &args.out {
        write_csv(path, &verified);
        eprintln!("[mint] wrote {path}");
    }
}
