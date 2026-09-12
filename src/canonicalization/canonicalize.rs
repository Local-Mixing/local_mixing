//! Deterministic polynomial ordering, bounded Rule-L search and counters.
pub use super::legacy_environment::{canon_monomial_cap, canon_rule_l_branch_cap};
use super::options::*;
use super::polynomial::*;
use crate::circuit::Permutation;
use rustc_hash::FxHashSet as HashSet;
use std::cmp::Ordering as CmpOrdering;
use std::sync::atomic::{AtomicU64, Ordering};
use std::time::Instant;
pub static CANON4_CORE_TIME: AtomicU64 = AtomicU64::new(0);
pub static POLYCANON_CORE_TIME: AtomicU64 = AtomicU64::new(0);
pub static CANON_BENCH_CALLS: AtomicU64 = AtomicU64::new(0);
pub static CANON4_RULE_L_TIME: AtomicU64 = AtomicU64::new(0);
pub static CANON4_RULE_L_CALLS: AtomicU64 = AtomicU64::new(0);
pub static CANON4_RULE_L_BRANCHES: AtomicU64 = AtomicU64::new(0);

/// Canonicalization-for-lookup calls skipped because a window touches more
/// than 64 distinct wires. `Monomial` is a `u64`, so such a window cannot be
/// represented without aliasing variables and potentially producing a false
/// database hit.
pub static OVERSIZED_CANON_SKIPS: AtomicU64 = AtomicU64::new(0);

/// Windows skipped while constructing polynomials because
/// `CANON_MONOMIAL_CAP` was exceeded.
pub static CANON_CAP_SKIPS: AtomicU64 = AtomicU64::new(0);

/// Windows skipped because Rule-L backtracking exceeded
/// `CANON_RULE_L_BRANCH_CAP`.
pub static CANON_RULE_L_SKIPS: AtomicU64 = AtomicU64::new(0);

/// Initial ranking method
/// Degree counts of a polynomial: [count_of_max_possible_deg, ..., count_of_deg_0]
/// Padded to max_possible_degree+1 entries so Vec comparison is always over equal-length
/// vectors and correctly ranks e.g. one degree-2 monomial above two degree-1 monomials.
fn degree_counts(poly: &Polynomial, max_possible_degree: usize) -> Vec<usize> {
    let mut counts = vec![0usize; max_possible_degree + 1];
    for m in poly {
        let deg = m.count_ones() as usize;
        counts[deg] += 1;
    }
    counts.reverse();
    counts
}

// Static accumulators for time spent in each rule (in nanoseconds)

// Given two orderings that produce the same canonical form,
// build the automorphism: sigma[order_a[pos]] = order_b[pos]
fn automorphism_from_orders(order_a: &[usize], order_b: &[usize], n: usize) -> Vec<usize> {
    let mut sigma = vec![0usize; n];
    for pos in 0..order_a.len() {
        sigma[order_a[pos]] = order_b[pos];
    }
    sigma
}

/// After canonicalization, trim trailing polynomials that are uninformative.
/// Starting from the last polynomial, remove P_i if both conditions hold:
///   1. P_i is trivial — its only monomial is the single variable x_i (bitmask 1 << i)
///   2. x_i does not appear in any other polynomial in the full list
/// Stop as soon as we reach a P_i that is non-trivial OR whose variable x_i
/// appears in some other polynomial. Keep everything from that point forward.
///
/// Returns the trimmed polynomial list. The permutation is left unchanged.
pub fn trim_canonicalized(polynomials: Vec<Polynomial>) -> Vec<Polynomial> {
    let n = polynomials.len();
    let mut keep_up_to = n; // exclusive upper bound — trim everything at or after this

    for i in (0..n).rev() {
        let bit = 1u64 << i;

        // Check if P_i is trivial: exactly one monomial which is just x_i
        let is_trivial =
            polynomials[i].len() == 1 && polynomials[i].iter().next().copied().unwrap() == bit;

        if !is_trivial {
            // Non-trivial polynomial — stop trimming here
            break;
        }

        // Check if x_i appears in any other polynomial (including higher degree monomials)
        let used_elsewhere = polynomials
            .iter()
            .enumerate()
            .any(|(j, poly)| j != i && poly.iter().any(|&m| m & bit != 0));

        if used_elsewhere {
            // x_i is referenced by another polynomial — stop trimming here
            break;
        }

        // P_i is trivial and x_i is unused elsewhere — trim it
        keep_up_to = i;
    }

    polynomials[..keep_up_to].to_vec()
}

const MONOMIAL_RANK_KEY_LEN_4: usize = 65;

const MONOMIAL_RANK_PREFIX_LEN_4: usize = 16;

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
struct MonomialRankKey4 {
    degree: u8,
    /// Big-endian packing of `encoded_ranks[..16]`. Comparing prefixes as
    /// integers equals comparing the first 16 bytes lexicographically, so the
    /// full 65-byte compare only runs on prefix ties (rare: it requires two
    /// monomials agreeing on their 16 highest-priority rank slots).
    prefix: u128,
    encoded_ranks: [u8; MONOMIAL_RANK_KEY_LEN_4],
}

impl Ord for MonomialRankKey4 {
    fn cmp(&self, other: &Self) -> CmpOrdering {
        // Identical ordering to `encoded_ranks.cmp(&other.encoded_ranks)`:
        // lexicographic byte compare == big-endian integer compare on the
        // packed prefix, and the tail settles prefix ties.
        self.prefix.cmp(&other.prefix).then_with(|| {
            self.encoded_ranks[MONOMIAL_RANK_PREFIX_LEN_4..]
                .cmp(&other.encoded_ranks[MONOMIAL_RANK_PREFIX_LEN_4..])
        })
    }
}

impl PartialOrd for MonomialRankKey4 {
    fn partial_cmp(&self, other: &Self) -> Option<CmpOrdering> {
        Some(self.cmp(other))
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
struct MonomialLevelKey4 {
    rank_key: MonomialRankKey4,
    coeff: usize,
}

type LevelEntry4 = (Monomial, usize, MonomialLevelKey4);

fn monomial_rank_key_4(m: Monomial, vr: &[usize], _n: usize) -> MonomialRankKey4 {
    let mut encoded_ranks = [0u8; MONOMIAL_RANK_KEY_LEN_4];
    let mut degree = 0usize;
    let mut mm = m;
    while mm != 0 {
        let v = mm.trailing_zeros() as usize;
        debug_assert!(vr[v] < u8::MAX as usize);
        encoded_ranks[degree] = (vr[v] + 1) as u8;
        degree += 1;
        mm &= mm - 1;
    }
    encoded_ranks[..degree].sort_unstable();
    let prefix = u128::from_be_bytes(
        encoded_ranks[..MONOMIAL_RANK_PREFIX_LEN_4]
            .try_into()
            .unwrap(),
    );
    MonomialRankKey4 {
        degree: degree as u8,
        prefix,
        encoded_ranks,
    }
}

fn monomial_level_key_4(m: Monomial, coeff: usize, vr: &[usize], n: usize) -> MonomialLevelKey4 {
    MonomialLevelKey4 {
        rank_key: monomial_rank_key_4(m, vr, n),
        coeff,
    }
}

fn cmp_level_key_4(a: &MonomialLevelKey4, b: &MonomialLevelKey4) -> CmpOrdering {
    b.rank_key
        .degree
        .cmp(&a.rank_key.degree)
        .then_with(|| a.rank_key.cmp(&b.rank_key))
        .then_with(|| b.coeff.cmp(&a.coeff))
}

// ── Compact level entries ────────────────────────────────────────────────────
// Valid iff every monomial degree <= MONOMIAL_RANK_PREFIX_LEN_4 (16): the fat
// key's 65-byte tail is then all zeros for every entry, so the full comparison
// collapses to (degree desc, prefix asc, coeff desc) — the same total order
// and the same equalities as the fat key. 32 bytes instead of ~128: 4x less
// build traffic, 4x smaller sort moves, and a 21-byte equality compare in the
// level walk. `canonicalize_polys_4` checks the degree bound once per top
// call (`compact_ok`); class polys and D-class polys reuse the same
// monomials, so that one check covers every scan in the recursion. Measured
// on the DB-armed fmix recipe the bound always holds; the fat path stays as
// the deg>16 fallback.

#[derive(Clone, Copy)]
struct LevelEntryC {
    prefix: u128,
    m: Monomial,
    coeff: u32,
    degree: u8,
}

// Shared rank packing for the compact paths: (degree, big-endian prefix of
// the +1-encoded, ascending-sorted rank bytes). Identical to
// monomial_rank_key_4 restricted to the first 16 slots; callers must uphold
// degree <= 16 (`compact_ok`).
#[inline]
fn rank_prefix_c(m: Monomial, vr: &[usize]) -> (u8, u128) {
    let mut ranks = [0u8; MONOMIAL_RANK_PREFIX_LEN_4];
    let mut degree = 0usize;
    let mut mm = m;
    while mm != 0 {
        let v = mm.trailing_zeros() as usize;
        debug_assert!(vr[v] < u8::MAX as usize);
        ranks[degree] = (vr[v] + 1) as u8;
        degree += 1;
        mm &= mm - 1;
    }
    ranks[..degree].sort_unstable();
    (degree as u8, u128::from_be_bytes(ranks))
}

#[inline]
fn level_entry_c(m: Monomial, coeff: usize, vr: &[usize]) -> LevelEntryC {
    let (degree, prefix) = rank_prefix_c(m, vr);
    LevelEntryC {
        prefix,
        m,
        // Class-poly coefficients count contributing wires, so they never
        // exceed n <= 64.
        coeff: coeff as u32,
        degree,
    }
}

#[inline]
fn cmp_level_c(a: &LevelEntryC, b: &LevelEntryC) -> CmpOrdering {
    b.degree
        .cmp(&a.degree)
        .then_with(|| a.prefix.cmp(&b.prefix))
        .then_with(|| b.coeff.cmp(&a.coeff))
}

#[inline]
fn eq_level_c(a: &LevelEntryC, b: &LevelEntryC) -> bool {
    a.degree == b.degree && a.prefix == b.prefix && a.coeff == b.coeff
}

fn sorted_level_entries_c(cp: &[(Monomial, usize)], vr: &[usize], entries: &mut Vec<LevelEntryC>) {
    entries.clear();
    entries.extend(cp.iter().map(|&(m, c)| level_entry_c(m, c, vr)));
    entries.sort_by(cmp_level_c);
}

fn sorted_level_entries_4(
    cp: &[(Monomial, usize)],
    vr: &[usize],
    n: usize,
    entries: &mut Vec<LevelEntry4>,
) {
    entries.clear();
    entries.extend(
        cp.iter()
            .map(|&(m, c)| (m, c, monomial_level_key_4(m, c, vr, n))),
    );
    entries.sort_by(|a, b| cmp_level_key_4(&a.2, &b.2));
}

// Count how many monomials in a level each wire appears in.
fn wire_freq_4(level: &[LevelEntry4], n: usize, freq: &mut Vec<usize>) {
    freq.resize(n, 0);
    freq.fill(0);
    for &(m, _, _) in level {
        let mut mm = m;
        while mm != 0 {
            freq[mm.trailing_zeros() as usize] += 1;
            mm &= mm - 1;
        }
    }
}

// Freq counts restricted to tied wires — the only entries split decisions
// read (untied wires never belong to a multi-member rank group). Masking
// before the bit walk skips most of the per-monomial popcount work.
fn wire_freq_tied_4(level: &[LevelEntry4], tied_mask: u64, n: usize, freq: &mut Vec<usize>) {
    freq.resize(n, 0);
    freq.fill(0);
    for &(m, _, _) in level {
        let mut mm = m & tied_mask;
        while mm != 0 {
            freq[mm.trailing_zeros() as usize] += 1;
            mm &= mm - 1;
        }
    }
}

fn wire_freq_c(level: &[LevelEntryC], n: usize, freq: &mut Vec<usize>) {
    freq.resize(n, 0);
    freq.fill(0);
    for e in level {
        let mut mm = e.m;
        while mm != 0 {
            freq[mm.trailing_zeros() as usize] += 1;
            mm &= mm - 1;
        }
    }
}

fn wire_freq_tied_c(level: &[LevelEntryC], tied_mask: u64, n: usize, freq: &mut Vec<usize>) {
    freq.resize(n, 0);
    freq.fill(0);
    for e in level {
        let mut mm = e.m & tied_mask;
        while mm != 0 {
            freq[mm.trailing_zeros() as usize] += 1;
            mm &= mm - 1;
        }
    }
}

// Split the FIRST (highest-priority) tied wire group whose members have different
// frequencies. Higher frequency → higher priority (lower rank number).
// Returns the split group's wire bitmask (None = no split) so the caller can
// track which class polys the split can possibly affect (cleanskip).
fn split_by_freq_4(
    vr: &mut Vec<usize>,
    n: usize,
    freq: &[usize],
    tied: &mut Vec<usize>,
    sorted: &mut Vec<usize>,
    sub_ranks: &mut Vec<usize>,
) -> Option<u64> {
    let max_rank = *vr.iter().max().unwrap_or(&0);
    for cur_rank in 0..=max_rank {
        tied.clear();
        tied.extend((0..n).filter(|&v| vr[v] == cur_rank));
        if tied.len() <= 1 {
            continue;
        }
        let first_freq = freq[tied[0]];
        if tied.iter().all(|&v| freq[v] == first_freq) {
            continue;
        }

        sorted.clear();
        sorted.extend_from_slice(tied);
        sorted.sort_by(|&a, &b| freq[b].cmp(&freq[a]));

        let mut sub_rank = 0usize;
        sub_ranks.clear();
        sub_ranks.resize(sorted.len(), 0);
        for i in 1..sorted.len() {
            if freq[sorted[i]] != freq[sorted[i - 1]] {
                sub_rank += 1;
            }
            sub_ranks[i] = sub_rank;
        }
        for v in 0..n {
            if vr[v] > cur_rank {
                vr[v] += sub_rank;
            }
        }
        for (i, &v) in sorted.iter().enumerate() {
            vr[v] = cur_rank + sub_ranks[i];
        }
        let group_mask = if n <= 64 {
            sorted.iter().fold(0u64, |acc, &v| acc | (1u64 << v))
        } else {
            // Degenerate width: report "could affect anything" so cleanskip
            // conservatively clears every flag.
            u64::MAX
        };
        return Some(group_mask);
    }
    None
}

// Tied groups precomputed once per master iteration (vr is constant between
// splits), stored flat to avoid allocs. groups_meta holds
// (rank, member bitmask, members start, members end) in ascending rank order,
// multi-member groups only. n <= 64 callers only.
fn tied_groups_4(
    vr: &[usize],
    n: usize,
    groups_meta: &mut Vec<(usize, u64, usize, usize)>,
    groups_members: &mut Vec<usize>,
) {
    groups_meta.clear();
    groups_members.clear();
    debug_assert!(n <= 64);
    let mut count = [0u8; 64];
    for &r in vr {
        count[r] += 1;
    }
    for r in 0..n {
        if count[r] > 1 {
            let start = groups_members.len();
            let mut mask = 0u64;
            for v in 0..n {
                if vr[v] == r {
                    groups_members.push(v);
                    mask |= 1u64 << v;
                }
            }
            groups_meta.push((r, mask, start, groups_members.len()));
        }
    }
}

// split_by_freq_4 with the rank iteration replaced by the precomputed group
// list. Identical selection: groups visited in ascending rank order; a group
// whose mask misses the level union has all-zero freqs (the old code also
// skipped it as all-equal); the stable freq-desc sort sees members in
// ascending wire id exactly like the old (0..n).filter build. Returns the
// split group's mask.
#[allow(clippy::too_many_arguments)]
fn split_by_freq_groups_4(
    vr: &mut Vec<usize>,
    n: usize,
    freq: &[usize],
    groups_meta: &[(usize, u64, usize, usize)],
    groups_members: &[usize],
    level_union: u64,
    sorted: &mut Vec<usize>,
    sub_ranks: &mut Vec<usize>,
) -> Option<u64> {
    for &(cur_rank, gmask, s, e) in groups_meta {
        if gmask & level_union == 0 {
            continue;
        }
        let members = &groups_members[s..e];
        let f0 = freq[members[0]];
        if members.iter().all(|&v| freq[v] == f0) {
            continue;
        }
        sorted.clear();
        sorted.extend_from_slice(members);
        sorted.sort_by(|&a, &b| freq[b].cmp(&freq[a]));
        let mut sub_rank = 0usize;
        sub_ranks.clear();
        sub_ranks.resize(sorted.len(), 0);
        for i in 1..sorted.len() {
            if freq[sorted[i]] != freq[sorted[i - 1]] {
                sub_rank += 1;
            }
            sub_ranks[i] = sub_rank;
        }
        for v in 0..n {
            if vr[v] > cur_rank {
                vr[v] += sub_rank;
            }
        }
        for (i, &v) in sorted.iter().enumerate() {
            vr[v] = cur_rank + sub_ranks[i];
        }
        return Some(gmask);
    }
    None
}

// Remapped polynomial key for tiebreak #1: replace each variable with its var_rank,
// sort ranks within each monomial, then sort monomials (highest priority first).
fn poly_key_4(poly: &Polynomial, vr: &[usize], n: usize) -> Vec<MonomialRankKey4> {
    let mut terms: Vec<MonomialRankKey4> = poly
        .iter()
        .map(|&m| monomial_rank_key_4(m, vr, n))
        .collect();
    terms.sort_by(|a, b| b.degree.cmp(&a.degree).then(a.cmp(b)));
    terms
}

// Compact tiebreak-#1 key, exact under the same deg <= 16 precondition as
// LevelEntryC: with the 65-byte tail all zeros, the fat rank-key ordering is
// its prefix ordering, so (degree, prefix) pairs sorted by
// (degree desc, prefix asc) reproduce poly_key_4's term order.
fn poly_key_c(poly: &Polynomial, vr: &[usize]) -> Vec<(u8, u128)> {
    let mut terms: Vec<(u8, u128)> = poly.iter().map(|&m| rank_prefix_c(m, vr)).collect();
    terms.sort_by(|a, b| b.0.cmp(&a.0).then(a.1.cmp(&b.1)));
    terms
}

// Keyed-wire comparison matching Vec<MonomialRankKey4>::cmp under the
// compact precondition: element order is prefix order (the fat Ord ignores
// the degree byte and the tail is zero), and equal prefixes imply equal
// degrees (a degree-d key has exactly d nonzero prefix bytes), so
// lexicographic-by-prefix plus the length tiebreak is the identical total
// order — and its Equal class is the identical equality.
fn cmp_poly_key_c(a: &[(u8, u128)], b: &[(u8, u128)]) -> CmpOrdering {
    let common = a.len().min(b.len());
    for i in 0..common {
        match a[i].1.cmp(&b[i].1) {
            CmpOrdering::Equal => {}
            other => return other,
        }
    }
    a.len().cmp(&b.len())
}

fn push_flat_canonical_form_4(
    polynomials: &[Polynomial],
    final_order: &[usize],
    wire_to_pos: &mut Vec<usize>,
    monomials: &mut Vec<Monomial>,
    out: &mut Vec<Option<Monomial>>,
) {
    let n = polynomials.len();
    wire_to_pos.resize(n, 0);
    for (pos, &wire) in final_order.iter().enumerate() {
        wire_to_pos[wire] = pos;
    }

    out.clear();
    for &wire in final_order {
        monomials.clear();
        monomials.extend(polynomials[wire].iter().map(|&m| {
            let mut r = 0u64;
            let mut mm = m;
            while mm != 0 {
                r |= 1u64 << wire_to_pos[mm.trailing_zeros() as usize];
                mm &= mm - 1;
            }
            r
        }));
        monomials.sort_unstable();
        out.extend(monomials.iter().copied().map(Some));
        out.push(None);
    }
}

// `groups`: the (groups_meta, groups_members) tied-group precompute for the
// n <= 64 fast path; None keeps the legacy per-level rank rescan (n > 64).
// Returns the split group's wire mask, None when no split fired.
#[allow(clippy::too_many_arguments)]
fn scan_class_poly_levels_4(
    cp: &[(Monomial, usize)],
    vr: &mut Vec<usize>,
    n: usize,
    tied_mask: u64,
    level_entries: &mut Vec<LevelEntry4>,
    freq: &mut Vec<usize>,
    tied: &mut Vec<usize>,
    sorted: &mut Vec<usize>,
    sub_ranks: &mut Vec<usize>,
    groups: Option<(&[(usize, u64, usize, usize)], &[usize])>,
) -> Option<u64> {
    sorted_level_entries_4(cp, vr, n, level_entries);
    let mut start = 0usize;
    while start < level_entries.len() {
        let mut end = start + 1;
        let mut level_union = level_entries[start].0;
        while end < level_entries.len() && level_entries[end].2 == level_entries[start].2 {
            level_union |= level_entries[end].0;
            end += 1;
        }
        // A level containing no tied wire gives every member of every tied
        // group frequency 0, so no split can fire — skip the freq count.
        if level_union & tied_mask != 0 {
            let split = if let Some((gm, gmem)) = groups {
                wire_freq_tied_4(&level_entries[start..end], tied_mask, n, freq);
                split_by_freq_groups_4(vr, n, freq, gm, gmem, level_union, sorted, sub_ranks)
            } else {
                wire_freq_4(&level_entries[start..end], n, freq);
                split_by_freq_4(vr, n, freq, tied, sorted, sub_ranks)
            };
            if split.is_some() {
                return split;
            }
        }
        start = end;
    }
    None
}

// Compact-entry scan: byte-for-byte the same level grouping and split
// decisions as scan_class_poly_levels_4 whenever all degrees <= 16
// (`compact_ok`).
#[allow(clippy::too_many_arguments)]
fn scan_class_poly_levels_c(
    cp: &[(Monomial, usize)],
    vr: &mut Vec<usize>,
    n: usize,
    tied_mask: u64,
    entries: &mut Vec<LevelEntryC>,
    freq: &mut Vec<usize>,
    tied: &mut Vec<usize>,
    sorted: &mut Vec<usize>,
    sub_ranks: &mut Vec<usize>,
    groups: Option<(&[(usize, u64, usize, usize)], &[usize])>,
) -> Option<u64> {
    sorted_level_entries_c(cp, vr, entries);
    let mut start = 0usize;
    while start < entries.len() {
        let mut end = start + 1;
        let mut level_union = entries[start].m;
        while end < entries.len() && eq_level_c(&entries[end], &entries[start]) {
            level_union |= entries[end].m;
            end += 1;
        }
        if level_union & tied_mask != 0 {
            let split = if let Some((gm, gmem)) = groups {
                wire_freq_tied_c(&entries[start..end], tied_mask, n, freq);
                split_by_freq_groups_4(vr, n, freq, gm, gmem, level_union, sorted, sub_ranks)
            } else {
                wire_freq_c(&entries[start..end], n, freq);
                split_by_freq_4(vr, n, freq, tied, sorted, sub_ranks)
            };
            if split.is_some() {
                return split;
            }
        }
        start = end;
    }
    None
}

// Bitmask of wires whose current rank is shared with another wire. Callers
// only use this to skip scans that provably cannot split anything, so for the
// (unreachable in practice) n > 64 case it degrades to "skip nothing".
fn tied_mask_4(vr: &[usize]) -> u64 {
    let n = vr.len();
    if n > 64 {
        return u64::MAX;
    }
    let mut count = [0u8; 64];
    for &r in vr {
        count[r] += 1;
    }
    let mut mask = 0u64;
    for (v, &r) in vr.iter().enumerate() {
        if count[r] > 1 {
            mask |= 1u64 << v;
        }
    }
    mask
}

fn has_ties_4(vr: &[usize]) -> bool {
    // Rank values stay contiguous in 0..n, so a bitmask detects duplicates in
    // one pass for the common n <= 64 case (monomials are u64 bitmasks, so n
    // never exceeds 64 on the polynomial-canonicalization paths).
    if vr.len() <= 64 {
        let mut seen = 0u64;
        for &r in vr {
            let bit = 1u64 << r;
            if seen & bit != 0 {
                return true;
            }
            seen |= bit;
        }
        return false;
    }
    let n = vr.len();
    (0..n).any(|v| (0..n).any(|u| u != v && vr[u] == vr[v]))
}

// Sort concatenated (monomial, 1) pairs and merge duplicates into counts.
// Produces the same monomial-ascending (monomial, count) sequence a BTreeMap
// build would, without per-insert tree traversal.
fn coalesce_class_poly(sum: &mut Vec<(Monomial, usize)>) {
    sum.sort_unstable_by_key(|&(m, _)| m);
    let mut write = 0usize;
    for read in 0..sum.len() {
        if write > 0 && sum[write - 1].0 == sum[read].0 {
            sum[write - 1].1 += sum[read].1;
        } else {
            sum[write] = sum[read];
            write += 1;
        }
    }
    sum.truncate(write);
}

// Is `b` reachable from `a` within `candidates` under known automorphisms that
// preserve the current rank coloring? Mirrors `is_same_orbit` (Rule L of the
// legacy canonicalizer); the coloring filter is what makes automorphisms
// discovered elsewhere in the recursion safe to reuse at this node, and each
// usable automorphism is applied in both directions (the group is closed
// under inversion).
// Does the orbit of `w` within `candidates` under `usable` (a set of
// coloring-preserving automorphisms, closed under inversion) contain any
// already-tried candidate? Because `usable` contains every inverse,
// reachability is symmetric, so this is exactly "exists t in tried reachable
// from t to w" — the historical per-(tried, candidate) check — computed with
// one BFS per candidate instead of one per pair. The caller maintains
// `usable` incrementally (vr is constant across one candidate loop).
fn canon4_orbit_hits_tried(
    w: usize,
    tried: &[usize],
    candidates: &[usize],
    usable: &[Vec<usize>],
) -> bool {
    if usable.is_empty() || tried.is_empty() {
        return false;
    }
    let cset: HashSet<usize> = candidates.iter().copied().collect();
    let tset: HashSet<usize> = tried.iter().copied().collect();
    let mut visited: HashSet<usize> = HashSet::default();
    let mut frontier = vec![w];
    visited.insert(w);
    while let Some(x) = frontier.pop() {
        for aut in usable {
            let img = aut[x];
            if tset.contains(&img) {
                return true;
            }
            if cset.contains(&img) && visited.insert(img) {
                frontier.push(img);
            }
        }
    }
    false
}

// Core loop: refine var_rank until fully resolved, then return final_order.
// `known_auts` accumulates automorphisms of the polynomial system discovered
// whenever two Rule L trials produce identical canonical forms; they are
// shared across the whole recursion to prune orbit-equivalent candidates.
// Per-invocation scratch buffers for canon4_run, pooled per thread and per
// recursion depth (each recursive call pops its own frame). Reuse kills the
// ~50+ heap allocations a fresh call would make; contents are always cleared
// before use, so pooling is invisible to the algorithm.
#[derive(Default)]
struct Canon4Frame {
    level_entries: Vec<LevelEntry4>,
    entries_c: Vec<LevelEntryC>,
    groups_meta: Vec<(usize, u64, usize, usize)>,
    groups_members: Vec<usize>,
    clean: Vec<bool>,
    freq_scratch: Vec<usize>,
    tied_scratch: Vec<usize>,
    sorted_scratch: Vec<usize>,
    sub_ranks_scratch: Vec<usize>,
    d_class_poly: Vec<(Monomial, usize)>,
    tied_buf: Vec<usize>,
    keyed_buf: Vec<(usize, Vec<MonomialRankKey4>)>,
    keyed_buf_c: Vec<(usize, Vec<(u8, u128)>)>,
    sub_ranks_buf: Vec<usize>,
    best_canonical: Vec<Option<Monomial>>,
    trial_canonical: Vec<Option<Monomial>>,
    canonical_monomials: Vec<Monomial>,
    wire_to_pos: Vec<usize>,
    tried: Vec<usize>,
    usable_auts: Vec<Vec<usize>>,
}

thread_local! {
    static CANON4_FRAMES: std::cell::RefCell<Vec<Canon4Frame>> =
        const { std::cell::RefCell::new(Vec::new()) };
}

#[allow(clippy::too_many_arguments)]
fn canon4_run(
    polynomials: &[Polynomial],
    class_polys: &[Vec<(Monomial, usize)>],
    class_unions: &[u64],
    vr: Vec<usize>,
    allow_rule_l: bool,
    known_auts: &mut Vec<Vec<usize>>,
    compact_ok: bool,
    search: &mut CanonSearchState<'_>,
) -> Result<Vec<usize>, ()> {
    let mut frame = CANON4_FRAMES
        .with(|pool| pool.borrow_mut().pop())
        .unwrap_or_default();
    let result = canon4_run_inner(
        polynomials,
        class_polys,
        class_unions,
        vr,
        allow_rule_l,
        known_auts,
        compact_ok,
        search,
        &mut frame,
    );
    CANON4_FRAMES.with(|pool| pool.borrow_mut().push(frame));
    result
}

#[allow(clippy::too_many_arguments)]
fn canon4_run_inner(
    polynomials: &[Polynomial],
    class_polys: &[Vec<(Monomial, usize)>],
    class_unions: &[u64],
    mut vr: Vec<usize>,
    allow_rule_l: bool,
    known_auts: &mut Vec<Vec<usize>>,
    compact_ok: bool,
    search: &mut CanonSearchState<'_>,
    frame: &mut Canon4Frame,
) -> Result<Vec<usize>, ()> {
    let n = polynomials.len();
    let Canon4Frame {
        level_entries,
        entries_c,
        groups_meta,
        groups_members,
        clean,
        freq_scratch,
        tied_scratch,
        sorted_scratch,
        sub_ranks_scratch,
        d_class_poly,
        tied_buf,
        keyed_buf,
        keyed_buf_c,
        sub_ranks_buf,
        best_canonical,
        trial_canonical,
        canonical_monomials,
        wire_to_pos,
        tried,
        usable_auts,
    } = frame;
    level_entries.clear();
    entries_c.clear();
    // One reservation per canon4_run: every scan reuses this buffer, and no
    // static class poly can exceed the largest one's length.
    let max_cp_len = class_polys.iter().map(|cp| cp.len()).max().unwrap_or(0);
    if compact_ok {
        entries_c.reserve(max_cp_len);
    } else {
        level_entries.reserve(max_cp_len);
    }
    freq_scratch.clear();
    tied_scratch.clear();
    sorted_scratch.clear();
    sub_ranks_scratch.clear();
    d_class_poly.clear();
    // The tied-group precompute and cleanskip apply on the n <= 64 fast path
    // only (monomials are u64 bitmasks, so canonicalization callers never
    // exceed it; the degenerate n > 64 path keeps the legacy per-level rank
    // rescan and never skips).
    let fast = n <= 64;
    // Cleanskip state is per canon4_run invocation: clean[i] = the last scan
    // of class poly i in THIS run returned no-split and every split since had
    // a group disjoint from its union. A split renumbers ranks
    // order-isomorphically outside its own group, so such a rescan must
    // return no-split again; skipping it cannot change vr and the result
    // stays byte-identical.
    clean.clear();
    clean.resize(class_polys.len(), false);

    'master: loop {
        if !has_ties_4(&vr) {
            break;
        }

        // Wires still sharing a rank; scans over polys that touch none of
        // them can never split anything and are skipped wholesale.
        let tied_mask = tied_mask_4(&vr);
        if fast {
            tied_groups_4(&vr, n, groups_meta, groups_members);
        }
        let groups: Option<(&[(usize, u64, usize, usize)], &[usize])> = if fast {
            Some((groups_meta.as_slice(), groups_members.as_slice()))
        } else {
            None
        };

        // Phase 1: scan P_{C_i} monomial levels; split by wire frequency.
        // Any split of the first splittable group → restart.
        for (idx, (cp, &cp_union)) in class_polys.iter().zip(class_unions).enumerate() {
            if cp_union & tied_mask == 0 {
                continue;
            }
            // A clean poly's rescan is provably a no-split; skipping it
            // leaves vr untouched, so the outcome is byte-identical.
            if fast && clean[idx] {
                continue;
            }
            let scan_res = if compact_ok {
                scan_class_poly_levels_c(
                    cp,
                    &mut vr,
                    n,
                    tied_mask,
                    entries_c,
                    freq_scratch,
                    tied_scratch,
                    sorted_scratch,
                    sub_ranks_scratch,
                    groups,
                )
            } else {
                scan_class_poly_levels_4(
                    cp,
                    &mut vr,
                    n,
                    tied_mask,
                    level_entries,
                    freq_scratch,
                    tied_scratch,
                    sorted_scratch,
                    sub_ranks_scratch,
                    groups,
                )
            };
            match scan_res {
                Some(gmask) => {
                    for (j, &u) in class_unions.iter().enumerate() {
                        if u & gmask != 0 {
                            clean[j] = false;
                        }
                    }
                    continue 'master;
                }
                None => {
                    clean[idx] = true;
                }
            }
        }

        if !has_ties_4(&vr) {
            break;
        }

        // Tiebreak #1: for each tied group, compare remapped polynomial keys.
        // First group where keys differ → split and restart. The compact and
        // fat branches compute the identical stable sort and sub-rank
        // partition (cmp_poly_key_c == Vec<MonomialRankKey4>::cmp under
        // compact_ok, including its Equal classes).
        let max_rank = *vr.iter().max().unwrap_or(&0);
        for cur_rank in 0..=max_rank {
            tied_buf.clear();
            tied_buf.extend((0..n).filter(|&v| vr[v] == cur_rank));
            if tied_buf.len() <= 1 {
                continue;
            }

            let mut sub_rank = 0usize;
            sub_ranks_buf.clear();
            sub_ranks_buf.resize(tied_buf.len(), 0);
            if compact_ok {
                keyed_buf_c.clear();
                keyed_buf_c.extend(
                    tied_buf
                        .iter()
                        .map(|&v| (v, poly_key_c(&polynomials[v], &vr))),
                );
                keyed_buf_c.sort_by(|a, b| cmp_poly_key_c(&a.1, &b.1));
                for i in 1..keyed_buf_c.len() {
                    if keyed_buf_c[i - 1].1 != keyed_buf_c[i].1 {
                        sub_rank += 1;
                    }
                    sub_ranks_buf[i] = sub_rank;
                }
            } else {
                keyed_buf.clear();
                keyed_buf.extend(
                    tied_buf
                        .iter()
                        .map(|&v| (v, poly_key_4(&polynomials[v], &vr, n))),
                );
                keyed_buf.sort_by(|a, b| a.1.cmp(&b.1));
                for i in 1..keyed_buf.len() {
                    if keyed_buf[i - 1].1 != keyed_buf[i].1 {
                        sub_rank += 1;
                    }
                    sub_ranks_buf[i] = sub_rank;
                }
            }
            if sub_rank > 0 {
                // This split's group is exactly the tied group at cur_rank;
                // clear clean flags for every class poly it touches.
                let gmask = if n <= 64 {
                    tied_buf.iter().fold(0u64, |acc, &v| acc | (1u64 << v))
                } else {
                    u64::MAX
                };
                for (j, &u) in class_unions.iter().enumerate() {
                    if u & gmask != 0 {
                        clean[j] = false;
                    }
                }
                for v in 0..n {
                    if vr[v] > cur_rank {
                        vr[v] += sub_rank;
                    }
                }
                if compact_ok {
                    for (i, &(v, _)) in keyed_buf_c.iter().enumerate() {
                        vr[v] = cur_rank + sub_ranks_buf[i];
                    }
                } else {
                    for (i, &(v, _)) in keyed_buf.iter().enumerate() {
                        vr[v] = cur_rank + sub_ranks_buf[i];
                    }
                }
                continue 'master;
            }
        }

        if !has_ties_4(&vr) {
            break;
        }

        // Tiebreak #2: dynamic class polys P_{D_i} from current rank groups.
        // Apply same monomial-level scanning as Phase 1.
        let max_rank_val = *vr.iter().max().unwrap_or(&0);
        for rk in 0..=max_rank_val {
            d_class_poly.clear();
            let mut d_union = 0u64;
            for w in 0..n {
                if vr[w] == rk {
                    for &m in &polynomials[w] {
                        d_class_poly.push((m, 1usize));
                        d_union |= m;
                    }
                }
            }
            if d_class_poly.is_empty() || d_union & tied_mask == 0 {
                continue;
            }
            coalesce_class_poly(d_class_poly);
            let dscan_res = if compact_ok {
                scan_class_poly_levels_c(
                    d_class_poly,
                    &mut vr,
                    n,
                    tied_mask,
                    entries_c,
                    freq_scratch,
                    tied_scratch,
                    sorted_scratch,
                    sub_ranks_scratch,
                    groups,
                )
            } else {
                scan_class_poly_levels_4(
                    d_class_poly,
                    &mut vr,
                    n,
                    tied_mask,
                    level_entries,
                    freq_scratch,
                    tied_scratch,
                    sorted_scratch,
                    sub_ranks_scratch,
                    groups,
                )
            };
            if let Some(gmask) = dscan_res {
                for (j, &u) in class_unions.iter().enumerate() {
                    if u & gmask != 0 {
                        clean[j] = false;
                    }
                }
                continue 'master;
            }
        }

        // Rule L: try each wire in the first tied group as the sole winner.
        // Take the candidate that produces the lexicographically smallest canonical form.
        let tied_rank = if n <= 64 {
            (0..n)
                .filter(|&v| tied_mask & (1u64 << v) != 0)
                .map(|v| vr[v])
                .min()
        } else {
            (0..n)
                .filter(|&v| (0..n).filter(|&u| vr[u] == vr[v]).count() > 1)
                .map(|v| vr[v])
                .min()
        };

        if let Some(tr) = tied_rank {
            if !allow_rule_l {
                return Err(());
            }
            let candidates: Vec<usize> = (0..n).filter(|&v| vr[v] == tr).collect();
            // Charge each recursive node against this top-level call's budget.
            if let Some(cap) = search.branch_cap {
                search.branches_used = search.branches_used.saturating_add(candidates.len() as u64);
                if search.branches_used > cap {
                    return Err(());
                }
            }
            let rule_l_start = Instant::now();
            CANON4_RULE_L_CALLS.fetch_add(1, Ordering::Relaxed);
            CANON4_RULE_L_BRANCHES.fetch_add(candidates.len() as u64, Ordering::Relaxed);
            best_canonical.clear();
            trial_canonical.clear();
            canonical_monomials.clear();
            wire_to_pos.clear();
            let mut have_best = false;
            let mut best_order: Vec<usize> = Vec::new();
            tried.clear();
            // Coloring-compatible automorphisms (plus inverses) for this
            // node's vr, synced lazily as the recursion appends to
            // known_auts. vr is constant across the candidate loop, so the
            // usability filter never changes and each aut is examined once.
            usable_auts.clear();
            let mut auts_synced = 0usize;

            for &w in &candidates {
                // Orbit pruning: a candidate reachable from an already-tried
                // one via a coloring-preserving automorphism completes to the
                // same canonical form, and equal forms never replace the best
                // (strict `<` below), so skipping it leaves the result
                // byte-identical while collapsing factorial symmetric blowup.
                while auts_synced < known_auts.len() {
                    let aut = &known_auts[auts_synced];
                    if aut.iter().enumerate().all(|(v, &img)| vr[img] == vr[v]) {
                        let mut inv = vec![0usize; aut.len()];
                        for (v, &img) in aut.iter().enumerate() {
                            inv[img] = v;
                        }
                        usable_auts.push(aut.clone());
                        usable_auts.push(inv);
                    }
                    auts_synced += 1;
                }
                if canon4_orbit_hits_tried(w, &tried, &candidates, &usable_auts) {
                    continue;
                }

                let mut trial_vr = vr.clone();
                for v in 0..n {
                    if trial_vr[v] > tr {
                        trial_vr[v] += 1;
                    }
                }
                for &other in &candidates {
                    if other != w {
                        trial_vr[other] = tr + 1;
                    }
                }

                let trial_order = canon4_run(
                    polynomials,
                    class_polys,
                    class_unions,
                    trial_vr,
                    true,
                    known_auts,
                    compact_ok,
                    search,
                )?;
                push_flat_canonical_form_4(
                    polynomials,
                    &trial_order,
                    wire_to_pos,
                    canonical_monomials,
                    trial_canonical,
                );

                if !have_best {
                    best_canonical.clear();
                    best_canonical.extend_from_slice(&trial_canonical);
                    best_order = trial_order;
                    have_best = true;
                } else if trial_canonical == best_canonical {
                    // Two completions with the same canonical form yield an
                    // automorphism; record it for pruning everywhere in the
                    // recursion (usability is re-checked per node).
                    known_auts.push(automorphism_from_orders(&best_order, &trial_order, n));
                } else if trial_canonical < best_canonical {
                    best_canonical.clear();
                    best_canonical.extend_from_slice(&trial_canonical);
                    best_order = trial_order;
                }

                tried.push(w);
            }
            let rule_l_elapsed = rule_l_start.elapsed();
            CANON4_RULE_L_TIME.fetch_add(rule_l_elapsed.as_nanos() as u64, Ordering::Relaxed);
            if trace_enabled(search.options)
                && rule_l_elapsed.as_millis() >= trace_threshold_ms(search.options)
            {
                eprintln!(
                    "[compress-trace] slow rule_l n={} tied_rank={} branches={} elapsed_ms={}",
                    n,
                    tr,
                    candidates.len(),
                    rule_l_elapsed.as_millis()
                );
            }
            return Ok(best_order);
        }

        break;
    }

    let mut final_order: Vec<usize> = (0..n).collect();
    final_order.sort_by_key(|&w| (vr[w], w));
    Ok(final_order)
}

struct CanonSearchState<'a> {
    branch_cap: Option<u64>,
    branches_used: u64,
    options: Option<&'a CanonicalizationOptions>,
}

/// Historical entry point with independently cached process-environment controls.
pub fn canonicalize_polys_4(
    polynomials: Vec<Polynomial>,
    allow_rule_l: bool,
) -> Result<(Vec<Polynomial>, Permutation), ()> {
    canonicalize_polys_4_using(polynomials, allow_rule_l, None)
}

/// Canonicalize using only the supplied limits; no configuration environment is read.
pub fn canonicalize_polys_4_with_options(
    polynomials: Vec<Polynomial>,
    allow_rule_l: bool,
    options: &CanonicalizationOptions,
) -> Result<(Vec<Polynomial>, Permutation), ()> {
    canonicalize_polys_4_using(polynomials, allow_rule_l, Some(options))
}

pub(super) fn canonicalize_polys_4_using(
    mut polynomials: Vec<Polynomial>,
    allow_rule_l: bool,
    options: Option<&CanonicalizationOptions>,
) -> Result<(Vec<Polynomial>, Permutation), ()> {
    let n = polynomials.len();
    if n == 0 {
        return Ok((vec![], Permutation { data: vec![] }));
    }
    // Compact level keys are exact iff every rank key's 65-byte tail stays
    // zero, i.e. every monomial degree <= MONOMIAL_RANK_PREFIX_LEN_4 (16).
    // Class polys and D-class polys reuse these same monomials, so one
    // top-level check covers every scan in the recursion.
    let compact_ok = !polynomials.iter().any(|p| {
        p.iter()
            .any(|m| m.count_ones() > MONOMIAL_RANK_PREFIX_LEN_4 as u32)
    });
    // Preserve the legacy branch-cap read at this point, after the empty case.
    let mut search = CanonSearchState {
        branch_cap: options.map_or_else(canon_rule_l_branch_cap, |value| value.rule_l_branch_cap),
        branches_used: 0,
        options,
    };
    for poly in &mut polynomials {
        normalize_polynomial(poly);
    }
    let max_degree = n;

    // Group wires by degree profile; highest-profile group = P_{C_1}.
    let mut profiles: Vec<(usize, Vec<usize>)> = (0..n)
        .map(|i| (i, degree_counts(&polynomials[i], max_degree)))
        .collect();
    profiles.sort_by(|a, b| b.1.cmp(&a.1));

    let mut class_groups: Vec<Vec<usize>> = Vec::new();
    {
        let mut current = vec![profiles[0].0];
        for i in 1..profiles.len() {
            if profiles[i].1 == profiles[i - 1].1 {
                current.push(profiles[i].0);
            } else {
                class_groups.push(current.clone());
                current = vec![profiles[i].0];
            }
        }
        class_groups.push(current);
    }

    // Build P_{C_i}: sum of polynomials in each class group (natural-number
    // coefficients), as monomial-sorted (monomial, count) vectors.
    let class_polys: Vec<Vec<(Monomial, usize)>> = class_groups
        .iter()
        .map(|group| {
            let mut sum: Vec<(Monomial, usize)> = Vec::new();
            for &wire in group {
                sum.extend(polynomials[wire].iter().map(|&m| (m, 1usize)));
            }
            coalesce_class_poly(&mut sum);
            sum
        })
        .collect();

    // Static per-class-poly wire unions let canon4_run skip scans over polys
    // that touch no still-tied wire.
    let class_unions: Vec<u64> = class_polys
        .iter()
        .map(|cp| cp.iter().fold(0u64, |acc, &(m, _)| acc | m))
        .collect();

    // All wires start tied; canon4_run refines iteratively.
    let mut known_auts: Vec<Vec<usize>> = Vec::new();
    let final_order = canon4_run(
        &polynomials,
        &class_polys,
        &class_unions,
        vec![0usize; n],
        allow_rule_l,
        &mut known_auts,
        compact_ok,
        &mut search,
    )?;

    let mut wire_to_pos = vec![0usize; n];
    for (pos, &wire) in final_order.iter().enumerate() {
        wire_to_pos[wire] = pos;
    }
    let remap_monomial = |m: Monomial| -> Monomial {
        let mut result = 0u64;
        let mut mm = m;
        while mm != 0 {
            result |= 1u64 << wire_to_pos[mm.trailing_zeros() as usize];
            mm &= mm - 1;
        }
        result
    };
    let canonical: Vec<Polynomial> = final_order
        .iter()
        .map(|&wire| polynomial_from_terms(polynomials[wire].iter().map(|&m| remap_monomial(m))))
        .collect();
    let canonical = trim_canonicalized(canonical);
    Ok((canonical, Permutation { data: final_order }))
}

#[cfg(test)]
use super::{keys::*, window::dense_wire_map};
#[cfg(test)]
use crate::circuit::{CircuitSeq, Gate, U1024, cancel_adjacent_duplicates, lane_state_len};
#[cfg(test)]
use primitive_types::{U256 as u256, U512 as u512};
#[cfg(test)]
#[path = "../../tests/unit/canonicalization/g57/tests.rs"]
mod tests;
