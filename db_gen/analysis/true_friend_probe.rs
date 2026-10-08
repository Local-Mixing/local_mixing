//! Sampled TRUE-FRIEND probe over the v3 pools composite, three modes.
//!
//! Spellings C (m gates, the KEY) and D (l gates, the FRIEND) of one function
//! f, delta = minlen(f). Loop C·D* with junctions ID and P_f, prefix points
//! a_i (on C) and b_j (on D).
//!
//! CHECKS. Side windows of a circuit use the arithmetic sanctioned routes
//! (window itself, or around through the junctions and a delta-chord):
//!     claimed_side(i,i',m) = min(i'-i, i+delta+(m-i'), (m-i)+delta+i')
//! A circuit's SLACK is the largest leeway any of its windows needs
//! (found >= claimed - slack); slack 0 = chord-tight. Cross pairs
//! (0<i<m, 0<j<l) use RELATIVIZED routes priced with proven distances:
//!     claimed_cross(i,j) = min(pc_i+pd_j, sc_i+sd_j,
//!                              pc_i+delta+sd_j, sc_i+delta+pd_j)
//! where pc_i = proven upper bound on minlen(C[..i]) (never above i), etc.
//! For chord-tight circuits this equals the arithmetic four-route formula;
//! for a junky key it prices the junction routes honestly instead of
//! silently re-imposing tightness on the key.
//!
//! MODES (an ordered pair C -> D qualifies when):
//!   (a) strict:  slack(C)=0, slack(D)=0, every cross check needs leeway 0.
//!   (c) asym-strict: slack(D)=0 and cross needs leeway 0; the KEY C is
//!       unconstrained (non-tight windows may be replaced).
//!   (b) leeway:  slack(D) <= t and cross needs <= t, where t = 2 for
//!       m+l >= 9, t = 1 for m+l in 7..=8, else 0. Key unconstrained.
//!   (b') floor:  as (b), but D's windows of width <= 4 must be exact
//!       (leeway only on D's larger windows and on cross checks) -- guards
//!       against serving friends with visibly cancelling/fusing short junk.
//! Containments (ordered pairs): (a) c= (c) c= (b') c= (b).
//!
//! BUILD MODE (--build): one sequential pass over the ENTIRE pools composite
//! (parallel by key prefix), per-length reservoir sampling of at most
//! --member-cap members per (pool, gate count), at most --pair-cap random
//! unordered pairs per pool (pools of <= 64 members are exhaustive at the
//! defaults), emitting each qualifying ordered pair as a composite record
//!     xxh3_128(key circuit's canonical blob) || friend blob
//! into four fresh RocksDB stores (--out-a/c/b/bp), each finalized with the
//! standard composite format marker, compaction and completion manifest so
//! the existing audit / to-lmdb / frozen pipeline accepts them. The
//! identity-function pool is skipped.
//!
//! DISTANCE ORACLE (layered; the frozen regular store is a REPLACEMENT table
//! whose classes need not contain minimal spellings):
//!   * exhaustive exact tables for minlen <= 3 (all 1/2/3-gate words up to
//!     relabelling; a key in no table provably has minlen >= 4);
//!   * store members (both canonical directions) as shortcut witnesses.
//! A check whose threshold (claimed - leeway) is <= 4 is decided exactly;
//! above that a pass means "no shortcut witnessed" -- the replacement-db
//! adversary's own evidence. Fails are always sound.
//!
//! Strata: S1 size-2 pools; S2 6-gate cohorts (pool 3..=64); S3 7-gate
//! cohorts (pool 3..=4096); S4 mid pools 65..=4096; S5 --mega keys sampled
//! per length (identity pool excluded: delta = 0 degenerates the notion).
//!
//! Read-only on both stores.
//!
//! ```text
//! true_friend_probe POOLS_COMPOSITE FROZEN_REGULAR_DIR [--out DIR]
//!     [--s1 N] [--s2 N] [--s3 N] [--s4 N] [--mega-per-len N] [--seed N]
//!     [--self-test] [--no-mega] [--mega KEY_HEX]...
//! ```

use local_mixing::circuit::CircuitSeq;
use local_mixing::database::frozen::FrozenDb;
use local_mixing::db_generation::curated_full::{
    COMPOSITE_COMPLETE_MARKER, COMPOSITE_FORMAT_MARKER, FUNCTION_KEY_BYTES, split_composite_key,
};
use rand::Rng as _;
use rand::SeedableRng as _;
use rayon::prelude::*;
use rocksdb::{DB, Direction, IteratorMode, Options, WriteBatch};
use rustc_hash::{FxHashMap, FxHashSet};
use std::collections::BTreeMap;
use std::error::Error;
use std::io::Write as _;
use std::path::PathBuf;
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::{Arc, Mutex};
use xxhash_rust::xxh3::Xxh3;

type AnyResult<T> = Result<T, Box<dyn Error + Send + Sync>>;

const MAX_GATES: usize = 16;
/// Slack/need values above the largest usable leeway collapse to FAIL.
const FAIL: u8 = 3;

fn pair_leeway(total_gates: usize) -> u8 {
    if total_gates >= 9 {
        2
    } else if total_gates >= 7 {
        1
    } else {
        0
    }
}

// ---------------------------------------------------------------------------
// Oracle.
// ---------------------------------------------------------------------------

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum MinLen {
    Exact(u8),
    /// Provably >= 4; `witness` = shortest stored spelling (255 = none).
    AtLeast4 {
        witness: u8,
    },
}

struct Oracle {
    frozen: FrozenDb,
    identity_key: [u8; 16],
    single_key: [u8; 16],
    two_keys: FxHashSet<[u8; 16]>,
    three_keys: FxHashSet<[u8; 16]>,
    shards: Vec<Mutex<FxHashMap<[u8; 16], MinLen>>>,
    calls: AtomicU64,
    store_lookups: AtomicU64,
    table_contradictions: AtomicU64,
}

fn all_gates(wires: u16) -> Vec<[u16; 3]> {
    let mut out = Vec::new();
    for t in 0..wires {
        for a in 0..wires {
            for b in 0..wires {
                if t != a && t != b && a != b {
                    out.push([t, a, b]);
                }
            }
        }
    }
    out
}

fn key_of(gates: &[[u16; 3]]) -> Option<[u8; 16]> {
    CircuitSeq {
        gates: gates.to_vec(),
    }
    .canonicalize_polys_single_hashed(false)
    .0
}

fn min_member_gates(value: &[u8]) -> Option<u8> {
    let mut best: Option<u8> = None;
    let mut pos = 0usize;
    while pos < value.len() {
        let len = value[pos] as usize;
        pos += 1;
        if len == 0 || len % 3 != 0 || pos + len > value.len() {
            break;
        }
        let gates = (len / 3) as u8;
        best = Some(best.map_or(gates, |b| b.min(gates)));
        pos += len;
    }
    best
}

impl Oracle {
    fn new(frozen: FrozenDb) -> Self {
        let identity_key = key_of(&[[0, 1, 2], [0, 1, 2]]).expect("identity canonicalization");
        let single_key = key_of(&[[0, 1, 2]]).expect("single-gate canonicalization");
        let g6 = all_gates(6);
        let first = [0u16, 1, 2];
        let mut two_keys: FxHashSet<[u8; 16]> = FxHashSet::default();
        for g2 in &g6 {
            if *g2 == first {
                continue;
            }
            if let Some(key) = key_of(&[first, *g2]) {
                if key != identity_key {
                    two_keys.insert(key);
                }
            }
        }
        let g9 = all_gates(9);
        let three_keys: FxHashSet<[u8; 16]> = g6
            .par_iter()
            .map(|g2| {
                let mut local: FxHashSet<[u8; 16]> = FxHashSet::default();
                for g3 in &g9 {
                    if let Some(key) = key_of(&[first, *g2, *g3]) {
                        local.insert(key);
                    }
                }
                local
            })
            .reduce(FxHashSet::default, |mut a, b| {
                a.extend(b);
                a
            })
            .into_iter()
            .filter(|k| *k != identity_key && *k != single_key && !two_keys.contains(k))
            .collect();
        eprintln!(
            "[oracle] exact tables: two={} three={}",
            two_keys.len(),
            three_keys.len()
        );
        Self {
            frozen,
            identity_key,
            single_key,
            two_keys,
            three_keys,
            shards: (0..256).map(|_| Mutex::new(FxHashMap::default())).collect(),
            calls: AtomicU64::new(0),
            store_lookups: AtomicU64::new(0),
            table_contradictions: AtomicU64::new(0),
        }
    }

    fn classify(&self, key_fwd: [u8; 16], circuit: &CircuitSeq) -> MinLen {
        if key_fwd == self.identity_key {
            return MinLen::Exact(0);
        }
        if key_fwd == self.single_key {
            return MinLen::Exact(1);
        }
        if self.two_keys.contains(&key_fwd) {
            return MinLen::Exact(2);
        }
        if self.three_keys.contains(&key_fwd) {
            return MinLen::Exact(3);
        }
        self.store_lookups.fetch_add(1, Ordering::Relaxed);
        let (key_rev, _, _) = circuit.canonicalize_polys_single_hashed(true);
        let mut witness = 255u8;
        for key in [Some(key_fwd), key_rev].into_iter().flatten() {
            if let Some(found) = self
                .frozen
                .get_regular(&key)
                .as_deref()
                .and_then(min_member_gates)
            {
                witness = witness.min(found);
            }
        }
        if witness < 4 {
            self.table_contradictions.fetch_add(1, Ordering::Relaxed);
            witness = 4;
        }
        MinLen::AtLeast4 { witness }
    }

    fn minlen(&self, gates: &[[u16; 3]]) -> MinLen {
        self.calls.fetch_add(1, Ordering::Relaxed);
        if gates.is_empty() {
            return MinLen::Exact(0);
        }
        if gates.len() == 1 {
            return MinLen::Exact(1);
        }
        let circuit = CircuitSeq {
            gates: gates.to_vec(),
        };
        let (key_fwd, _, _) = circuit.canonicalize_polys_single_hashed(false);
        let Some(key_fwd) = key_fwd else {
            return MinLen::AtLeast4 { witness: 255 };
        };
        let shard = key_fwd[0] as usize;
        if let Some(&hit) = self.shards[shard].lock().unwrap().get(&key_fwd) {
            return hit;
        }
        let result = self.classify(key_fwd, &circuit);
        self.shards[shard].lock().unwrap().insert(key_fwd, result);
        result
    }

    /// Smallest leeway at which `minlen(gates) >= claimed - leeway` holds,
    /// capped at FAIL, plus whether that pass is witness-based (threshold
    /// above the exact-table horizon and covered only by store witnesses).
    fn needed_leeway(&self, gates: &[[u16; 3]], claimed: usize) -> (u8, bool) {
        match self.minlen(gates) {
            MinLen::Exact(d) => (
                (claimed.saturating_sub(d as usize)).min(FAIL as usize) as u8,
                false,
            ),
            MinLen::AtLeast4 { witness } => {
                // Passing exactly needs threshold <= 4; passing on the
                // witness needs threshold <= witness.
                let need_exact = claimed.saturating_sub(4);
                let need_witness = claimed.saturating_sub(witness as usize);
                if need_witness < need_exact {
                    ((need_witness).min(FAIL as usize) as u8, true)
                } else {
                    ((need_exact).min(FAIL as usize) as u8, false)
                }
            }
        }
    }

    /// Proven upper bound on minlen, never above the word's own length.
    fn upper(&self, gates: &[[u16; 3]]) -> u8 {
        let own = gates.len().min(255) as u8;
        match self.minlen(gates) {
            MinLen::Exact(d) => d.min(own),
            MinLen::AtLeast4 { witness } => witness.min(own),
        }
    }
}

fn claimed_side(i: usize, i2: usize, m: usize, delta: usize) -> usize {
    (i2 - i).min(i + delta + (m - i2)).min((m - i) + delta + i2)
}

// ---------------------------------------------------------------------------
// Per-circuit evaluation.
// ---------------------------------------------------------------------------

struct CircuitEval {
    /// Max leeway needed by windows of width <= 4 / width >= 5 (FAIL-capped).
    slack_small: u8,
    slack_large: u8,
    /// Any window pass at its needed level was witness-based.
    witnessed: bool,
    /// Proven upper bounds: prefix_min[i] ~ minlen(C[..i]), suffix_min[i] ~
    /// minlen(C[i..]), for i = 0..=m. Both never above the raw gate count.
    prefix_min: Vec<u8>,
    suffix_min: Vec<u8>,
}

fn eval_circuit(gates: &[[u16; 3]], delta: usize, oracle: &Oracle) -> CircuitEval {
    let m = gates.len();
    let mut slack_small = 0u8;
    let mut slack_large = 0u8;
    let mut witnessed = false;
    let mut prefix_min = vec![0u8; m + 1];
    let mut suffix_min = vec![0u8; m + 1];
    for i in 0..=m {
        prefix_min[i] = i.min(255) as u8;
        suffix_min[i] = (m - i).min(255) as u8;
    }
    for width in 2..=m {
        for i in 0..=(m - width) {
            let i2 = i + width;
            let window = &gates[i..i2];
            let claimed = claimed_side(i, i2, m, delta);
            let (need, wit) = oracle.needed_leeway(window, claimed);
            witnessed |= wit && need > 0;
            if width <= 4 {
                slack_small = slack_small.max(need);
            } else {
                slack_large = slack_large.max(need);
            }
            if i == 0 {
                prefix_min[i2] = prefix_min[i2].min(oracle.upper(window));
            }
            if i2 == m {
                suffix_min[i] = suffix_min[i].min(oracle.upper(window));
            }
        }
    }
    CircuitEval {
        slack_small,
        slack_large,
        witnessed,
        prefix_min,
        suffix_min,
    }
}

fn rename_extras(d: &[[u16; 3]], semantic_width: u16, fresh_base: u16) -> Vec<[u16; 3]> {
    let mut map: FxHashMap<u16, u16> = FxHashMap::default();
    let mut next = fresh_base;
    d.iter()
        .map(|&[t, a, b]| {
            let mut g = [t, a, b];
            for w in &mut g {
                if *w >= semantic_width {
                    *w = *map.entry(*w).or_insert_with(|| {
                        let v = next;
                        next += 1;
                        v
                    });
                }
            }
            g
        })
        .collect()
}

/// Max leeway needed over all cross checks (FAIL-capped, early abort past
/// FAIL), with relativized claims priced from proven prefix/suffix bounds.
fn cross_need(
    c: &[[u16; 3]],
    d: &[[u16; 3]],
    ce: &CircuitEval,
    de: &CircuitEval,
    delta: usize,
    oracle: &Oracle,
    checks: &mut u64,
) -> (u8, bool) {
    let m = c.len();
    let l = d.len();
    let mut order: Vec<(usize, usize, usize)> = Vec::with_capacity((m - 1) * (l - 1));
    for i in 1..m {
        for j in 1..l {
            let cost = (i + j).min((m - i) + (l - j));
            order.push((cost, i, j));
        }
    }
    order.sort_unstable();
    let mut need_max = 0u8;
    let mut witnessed = false;
    let mut word: Vec<[u16; 3]> = Vec::with_capacity(m + l);
    for (_, i, j) in order {
        let (pc, sc) = (ce.prefix_min[i] as usize, ce.suffix_min[i] as usize);
        let (pd, sd) = (de.prefix_min[j] as usize, de.suffix_min[j] as usize);
        let claimed = (pc + pd)
            .min(sc + sd)
            .min(pc + delta + sd)
            .min(sc + delta + pd);
        word.clear();
        if i + j <= (m - i) + (l - j) {
            word.extend(c[..i].iter().rev());
            word.extend_from_slice(&d[..j]);
        } else {
            word.extend_from_slice(&c[i..]);
            word.extend(d[j..].iter().rev());
        }
        *checks += 1;
        let (need, wit) = oracle.needed_leeway(&word, claimed);
        witnessed |= wit && need > 0;
        need_max = need_max.max(need);
        if need_max >= FAIL {
            return (FAIL, witnessed);
        }
    }
    (need_max, witnessed)
}

// ---------------------------------------------------------------------------
// Pool access.
// ---------------------------------------------------------------------------

struct Pool {
    key: [u8; FUNCTION_KEY_BYTES],
    members: Vec<Vec<[u16; 3]>>,
    truncated: bool,
}

fn read_pool(db: &DB, probe: &[u8], cap: usize) -> Option<Pool> {
    let mut it = db.iterator(IteratorMode::From(probe, Direction::Forward));
    let mut key: Option<[u8; FUNCTION_KEY_BYTES]> = None;
    let mut members = Vec::new();
    let mut truncated = false;
    for item in it.by_ref() {
        let Ok((record, _)) = item else { break };
        if record.as_ref() == COMPOSITE_FORMAT_MARKER {
            continue;
        }
        let Ok((k, blob)) = split_composite_key(&record) else {
            continue;
        };
        match key {
            None => key = Some(k),
            Some(existing) if existing != k => break,
            _ => {}
        }
        if members.len() >= cap {
            truncated = true;
            break;
        }
        members.push(CircuitSeq::from_blob(blob).gates);
    }
    key.map(|key| Pool {
        key,
        members,
        truncated,
    })
}

fn sample_mega(
    db: &DB,
    key: &[u8; FUNCTION_KEY_BYTES],
    lengths: &[usize],
    want: usize,
    rng: &mut rand::rngs::StdRng,
) -> BTreeMap<usize, Vec<Vec<[u16; 3]>>> {
    let mut out: BTreeMap<usize, Vec<Vec<[u16; 3]>>> = BTreeMap::new();
    let mut seen: FxHashSet<Vec<u8>> = FxHashSet::default();
    let need = |out: &BTreeMap<usize, Vec<Vec<[u16; 3]>>>| {
        lengths
            .iter()
            .any(|&l| out.get(&l).map_or(0, Vec::len) < want)
    };
    let mut attempts = 0usize;
    while need(&out) && attempts < want * lengths.len() * 200 {
        attempts += 1;
        let mut probe = key.to_vec();
        let extra = rng.random_range(1..=33usize) * 3;
        for _ in 0..extra {
            probe.push(rng.random());
        }
        let mut it = db.iterator(IteratorMode::From(&probe, Direction::Forward));
        let Some(Ok((record, _))) = it.next() else {
            continue;
        };
        let Ok((k, blob)) = split_composite_key(&record) else {
            continue;
        };
        if k != *key {
            continue;
        }
        let gates = blob.len() / 3;
        if !lengths.contains(&gates) {
            continue;
        }
        let entry = out.entry(gates).or_default();
        if entry.len() >= want || !seen.insert(blob.to_vec()) {
            continue;
        }
        entry.push(CircuitSeq::from_blob(blob).gates);
    }
    out
}

// ---------------------------------------------------------------------------
// Accounting.
// ---------------------------------------------------------------------------

#[derive(Default)]
struct PairCell {
    /// Unordered pairs evaluated.
    tested: u64,
    /// Mode (a): symmetric, counts unordered qualifying pairs.
    a_pass: u64,
    /// Modes (c)/(b)/(b'): ordered qualifying pairs (0, 1 or 2 per unordered).
    c_pass: u64,
    b_pass: u64,
    bp_pass: u64,
}

#[derive(Default)]
struct StratumStats {
    pools: u64,
    circuits: u64,
    circuits_tight: u64,
    /// b-qualifying ordered pairs by the leeway their worst check needed.
    b_need_hist: [u64; 3],
    checks: u64,
    cells: BTreeMap<(usize, usize), PairCell>,
    /// members with >= 1 (b)-friend, split by whether the KEY is tight.
    keys_served_tight: u64,
    keys_served_junky: u64,
    keys_total_tight: u64,
    keys_total_junky: u64,
    degree_rows: Vec<(String, usize, usize, usize, u64, u64, u64, u64)>,
}

impl StratumStats {
    fn merge(mut self, other: Self) -> Self {
        self.pools += other.pools;
        self.circuits += other.circuits;
        self.circuits_tight += other.circuits_tight;
        for i in 0..3 {
            self.b_need_hist[i] += other.b_need_hist[i];
        }
        self.checks += other.checks;
        for (k, v) in other.cells {
            let cell = self.cells.entry(k).or_default();
            cell.tested += v.tested;
            cell.a_pass += v.a_pass;
            cell.c_pass += v.c_pass;
            cell.b_pass += v.b_pass;
            cell.bp_pass += v.bp_pass;
        }
        self.keys_served_tight += other.keys_served_tight;
        self.keys_served_junky += other.keys_served_junky;
        self.keys_total_tight += other.keys_total_tight;
        self.keys_total_junky += other.keys_total_junky;
        self.degree_rows.extend(other.degree_rows);
        self
    }
}

fn hex(key: &[u8; FUNCTION_KEY_BYTES]) -> String {
    key.iter().map(|b| format!("{b:02x}")).collect()
}

/// Evaluate one pool's sampled members under all modes.
/// Returns per-member (b)-friend counts (as key).
#[allow(clippy::too_many_arguments)]
fn test_pool_pairs(
    members: &[Vec<[u16; 3]>],
    delta: usize,
    semantic_width: u16,
    oracle: &Oracle,
    stats: &mut StratumStats,
    pair_cap: usize,
    rng: &mut rand::rngs::StdRng,
) -> Vec<u64> {
    let n = members.len();
    let evals: Vec<Option<CircuitEval>> = members
        .iter()
        .map(|m| {
            if m.len() > MAX_GATES {
                return None;
            }
            stats.circuits += 1;
            let e = eval_circuit(m, delta, oracle);
            stats.checks += (m.len() * (m.len() + 1) / 2) as u64;
            if e.slack_small == 0 && e.slack_large == 0 {
                stats.circuits_tight += 1;
            }
            Some(e)
        })
        .collect();
    for e in evals.iter().flatten() {
        if e.slack_small == 0 && e.slack_large == 0 {
            stats.keys_total_tight += 1;
        } else {
            stats.keys_total_junky += 1;
        }
    }
    let mut b_degree = vec![0u64; n];
    let mut all_pairs: Vec<(usize, usize)> = (0..n)
        .flat_map(|a| ((a + 1)..n).map(move |b| (a, b)))
        .collect();
    if all_pairs.len() > pair_cap {
        for i in 0..pair_cap {
            let j = rng.random_range(i..all_pairs.len());
            all_pairs.swap(i, j);
        }
        all_pairs.truncate(pair_cap);
    }
    for (x, y) in all_pairs {
        let (Some(ex), Some(ey)) = (&evals[x], &evals[y]) else {
            continue;
        };
        let (lx, ly) = (members[x].len(), members[y].len());
        let total = lx + ly;
        let t = pair_leeway(total);
        let cell = stats.cells.entry((lx.min(ly), lx.max(ly))).or_default();
        cell.tested += 1;
        let slack = |e: &CircuitEval| e.slack_small.max(e.slack_large);
        // Cheap pre-screen: if neither ordering can qualify in any mode,
        // skip the cross work entirely.
        let x_friend_b = slack(ex) <= t;
        let y_friend_b = slack(ey) <= t;
        let a_possible = slack(ex) == 0 && slack(ey) == 0;
        if !x_friend_b && !y_friend_b && !a_possible {
            continue;
        }
        let fresh_base = members[x]
            .iter()
            .chain(members[y].iter())
            .flatten()
            .copied()
            .max()
            .unwrap_or(0)
            + 1;
        let d2 = rename_extras(&members[y], semantic_width, fresh_base);
        let (need, _wit) = cross_need(&members[x], &d2, ex, ey, delta, oracle, &mut stats.checks);
        if a_possible && need == 0 {
            cell.a_pass += 1;
        }
        // Ordered: key -> friend.
        for (_key_ix, friend_ix, fe) in [(x, y, ey), (y, x, ex)] {
            if need == 0 && slack(fe) == 0 {
                cell.c_pass += 1;
            }
            if need <= t && slack(fe) <= t {
                cell.b_pass += 1;
                b_degree[_key_ix] += 1;
                stats.b_need_hist[need.max(slack(fe)) as usize] += 1;
                if fe.slack_small == 0 && need <= t {
                    cell.bp_pass += 1;
                }
            }
            let _ = friend_ix;
        }
    }
    for (ix, e) in evals.iter().enumerate() {
        let Some(e) = e else { continue };
        if b_degree[ix] > 0 {
            if e.slack_small == 0 && e.slack_large == 0 {
                stats.keys_served_tight += 1;
            } else {
                stats.keys_served_junky += 1;
            }
        }
    }
    b_degree
}

fn semantic_width(members: &[Vec<[u16; 3]>]) -> u16 {
    members
        .iter()
        .min_by_key(|m| m.len())
        .map(|m| m.iter().flatten().copied().max().map_or(0, |w| w + 1))
        .unwrap_or(0)
}

fn pool_delta(oracle: &Oracle, members: &[Vec<[u16; 3]>]) -> Option<usize> {
    let shortest = members.iter().min_by_key(|m| m.len())?;
    Some(match oracle.minlen(shortest) {
        MinLen::Exact(d) => d as usize,
        MinLen::AtLeast4 { witness } => (witness as usize).min(shortest.len()),
    })
}

// ---------------------------------------------------------------------------
// Build mode: emit the four true-friend composite stores in one pass.
// ---------------------------------------------------------------------------

/// Build-mode stores, each an independently emittable pair predicate:
///   a  = both circuits tight, cross exact (symmetric strict)
///   c  = friend tight, cross exact, key unconstrained
///   b  = friend slack <= t, cross <= t (t by pair total length)
///   bp = b with the friend's width<=4 windows exact ("b-floor")
///   v1 = friend TIGHT, cross <= t          (spread leeway only)
///   v2 = bp-style friend, cross EXACT      (friend slack only)
///   c1 = bp with every leeway capped at 1  (uniformly milder)
const MODE_LABELS: [&str; 7] = ["a", "c", "b", "b-floor", "v1", "v2", "c1"];
const MODE_FLAGS: [&str; 7] = [
    "--out-a", "--out-c", "--out-b", "--out-bp", "--out-v1", "--out-v2", "--out-c1",
];
const MODE_COUNT: usize = 7;

struct BuildCounters {
    pools: AtomicU64,
    pools_identity: AtomicU64,
    members_seen: AtomicU64,
    members_sampled: AtomicU64,
    pairs_tested: AtomicU64,
    emitted: [AtomicU64; MODE_COUNT],
    done_prefixes: AtomicU64,
}

impl BuildCounters {
    fn new() -> Self {
        Self {
            pools: AtomicU64::new(0),
            pools_identity: AtomicU64::new(0),
            members_seen: AtomicU64::new(0),
            members_sampled: AtomicU64::new(0),
            pairs_tested: AtomicU64::new(0),
            emitted: std::array::from_fn(|_| AtomicU64::new(0)),
            done_prefixes: AtomicU64::new(0),
        }
    }
}

fn open_fresh_composite(path: &str) -> AnyResult<Arc<DB>> {
    if std::path::Path::new(path).exists() {
        return Err(format!("refusing to overwrite existing output {path}").into());
    }
    if let Some(parent) = std::path::Path::new(path).parent() {
        if !parent.as_os_str().is_empty() {
            std::fs::create_dir_all(parent)?;
        }
    }
    let mut options = Options::default();
    options.create_if_missing(true);
    options.set_compression_type(rocksdb::DBCompressionType::Zstd);
    options.set_bottommost_compression_type(rocksdb::DBCompressionType::Zstd);
    options.increase_parallelism(8);
    let db = DB::open(&options, path)?;
    db.put(COMPOSITE_FORMAT_MARKER, b"1")?;
    Ok(Arc::new(db))
}

/// Compaction + the standard composite completion manifest (keys, candidates,
/// xxh3-128 digest over `[u16 record len][record]` in key order, metadata
/// skipped) -- byte-compatible with `build_curated_full audit`.
fn finalize_output(db: &DB, label: &str) -> AnyResult<(u64, u64)> {
    db.flush()?;
    db.compact_range::<&[u8], &[u8]>(None, None);
    let mut keys = 0u64;
    let mut candidates = 0u64;
    let mut previous: Option<[u8; 16]> = None;
    let mut digest = Xxh3::new();
    for item in db.iterator(IteratorMode::Start) {
        let (record, _) = item?;
        if record.as_ref() == COMPOSITE_FORMAT_MARKER
            || record.as_ref() == COMPOSITE_COMPLETE_MARKER
        {
            continue;
        }
        let record_len = u16::try_from(record.len())?;
        digest.update(&record_len.to_le_bytes());
        digest.update(&record);
        let (key, _) = split_composite_key(&record)?;
        if previous != Some(key) {
            keys += 1;
            previous = Some(key);
        }
        candidates += 1;
    }
    let value = digest.digest128();
    let mut manifest = [0u8; 32];
    manifest[..8].copy_from_slice(&keys.to_le_bytes());
    manifest[8..16].copy_from_slice(&candidates.to_le_bytes());
    manifest[16..].copy_from_slice(&value.to_le_bytes());
    db.put(COMPOSITE_COMPLETE_MARKER, manifest)?;
    db.flush()?;
    eprintln!("[build:{label}] keys={keys} pairs={candidates} digest={value:032x}");
    Ok((keys, candidates))
}

/// Test one pool's sampled members and append qualifying ordered pairs to the
/// four stores. `sampled` holds raw blobs (store encoding); the emitted record
/// is `xxh3_128(key blob).to_le_bytes() || friend blob`.
fn process_pool_build(
    oracle: &Oracle,
    outs: &[Option<Arc<DB>>; MODE_COUNT],
    sampled: &[Vec<u8>],
    seen_total: u64,
    pair_cap: usize,
    rng: &mut rand::rngs::StdRng,
    counters: &BuildCounters,
) -> AnyResult<()> {
    counters.pools.fetch_add(1, Ordering::Relaxed);
    counters
        .members_seen
        .fetch_add(seen_total, Ordering::Relaxed);
    counters
        .members_sampled
        .fetch_add(sampled.len() as u64, Ordering::Relaxed);
    if sampled.len() < 2 {
        return Ok(());
    }
    let gates: Vec<Vec<[u16; 3]>> = sampled
        .iter()
        .map(|b| CircuitSeq::from_blob(b).gates)
        .collect();
    let width = semantic_width(&gates);
    let Some(delta) = pool_delta(oracle, &gates) else {
        return Ok(());
    };
    if delta == 0 {
        counters.pools_identity.fetch_add(1, Ordering::Relaxed);
        return Ok(());
    }
    let evals: Vec<Option<CircuitEval>> = gates
        .iter()
        .map(|m| (m.len() >= 2 && m.len() <= MAX_GATES).then(|| eval_circuit(m, delta, oracle)))
        .collect();
    let n = gates.len();
    let mut pairs: Vec<(usize, usize)> = (0..n)
        .flat_map(|a| ((a + 1)..n).map(move |b| (a, b)))
        .collect();
    if pairs.len() > pair_cap {
        for i in 0..pair_cap {
            let j = rng.random_range(i..pairs.len());
            pairs.swap(i, j);
        }
        pairs.truncate(pair_cap);
    }
    let mut batches: Vec<WriteBatch> = (0..MODE_COUNT).map(|_| WriteBatch::default()).collect();
    let mut checks = 0u64;
    for (x, y) in pairs {
        let (Some(ex), Some(ey)) = (&evals[x], &evals[y]) else {
            continue;
        };
        counters.pairs_tested.fetch_add(1, Ordering::Relaxed);
        let t = pair_leeway(gates[x].len() + gates[y].len());
        let slack = |e: &CircuitEval| e.slack_small.max(e.slack_large);
        let a_possible = slack(ex) == 0 && slack(ey) == 0;
        if slack(ex) > t && slack(ey) > t && !a_possible {
            continue;
        }
        let fresh_base = gates[x]
            .iter()
            .chain(gates[y].iter())
            .flatten()
            .copied()
            .max()
            .unwrap_or(0)
            + 1;
        let d2 = rename_extras(&gates[y], width, fresh_base);
        let (need, _wit) = cross_need(&gates[x], &d2, ex, ey, delta, oracle, &mut checks);
        for (key_ix, friend_ix, fe) in [(x, y, ey), (y, x, ex)] {
            let fslack = slack(fe);
            let t1 = t.min(1);
            let a_ok = a_possible && need == 0;
            let c_ok = fslack == 0 && need == 0;
            let b_ok = fslack <= t && need <= t;
            let bp_ok = b_ok && fe.slack_small == 0;
            let v1_ok = fslack == 0 && need <= t;
            let v2_ok = fslack <= t && fe.slack_small == 0 && need == 0;
            let c1_ok = fslack <= t1 && fe.slack_small == 0 && need <= t1;
            let oks = [a_ok, c_ok, b_ok, bp_ok, v1_ok, v2_ok, c1_ok];
            if !oks
                .iter()
                .zip(outs.iter())
                .any(|(&ok, out)| ok && out.is_some())
            {
                continue;
            }
            let key16 = xxhash_rust::xxh3::xxh3_128(&sampled[key_ix]).to_le_bytes();
            let mut record = Vec::with_capacity(16 + sampled[friend_ix].len());
            record.extend_from_slice(&key16);
            record.extend_from_slice(&sampled[friend_ix]);
            for (mode, ok) in oks.into_iter().enumerate() {
                if ok && outs[mode].is_some() {
                    batches[mode].put(&record, []);
                    counters.emitted[mode].fetch_add(1, Ordering::Relaxed);
                }
            }
        }
    }
    for (mode, batch) in batches.into_iter().enumerate() {
        if !batch.is_empty() {
            if let Some(out) = &outs[mode] {
                out.write(batch)?;
            }
        }
    }
    Ok(())
}

#[allow(clippy::too_many_arguments)]
fn build_prefix(
    pools: &DB,
    oracle: &Oracle,
    outs: &[Option<Arc<DB>>; MODE_COUNT],
    prefix: u8,
    member_cap: usize,
    pair_cap: usize,
    seed: u64,
    counters: &BuildCounters,
    started: &std::time::Instant,
) -> AnyResult<()> {
    let start = [prefix];
    let mut current: Option<[u8; 16]> = None;
    let mut lengths: BTreeMap<usize, (u64, Vec<Vec<u8>>)> = BTreeMap::new();
    let mut rng = rand::rngs::StdRng::seed_from_u64(seed ^ prefix as u64);
    let mut flush = |key: Option<[u8; 16]>,
                     lengths: &mut BTreeMap<usize, (u64, Vec<Vec<u8>>)>,
                     rng: &mut rand::rngs::StdRng|
     -> AnyResult<()> {
        let Some(_key) = key else { return Ok(()) };
        let seen: u64 = lengths.values().map(|(s, _)| *s).sum();
        let sampled: Vec<Vec<u8>> = lengths
            .values_mut()
            .flat_map(|(_, v)| std::mem::take(v))
            .collect();
        lengths.clear();
        process_pool_build(oracle, outs, &sampled, seen, pair_cap, rng, counters)
    };
    for item in pools.iterator(IteratorMode::From(&start, Direction::Forward)) {
        let Ok((record, _)) = item else { break };
        if record.is_empty() || record[0] != prefix {
            break;
        }
        if record.as_ref() == COMPOSITE_FORMAT_MARKER
            || record.as_ref() == COMPOSITE_COMPLETE_MARKER
        {
            continue;
        }
        let Ok((k, blob)) = split_composite_key(&record) else {
            continue;
        };
        if current != Some(k) {
            let previous = current.take();
            flush(previous, &mut lengths, &mut rng)?;
            current = Some(k);
            rng = rand::rngs::StdRng::seed_from_u64(
                seed ^ u64::from_le_bytes(k[..8].try_into().unwrap()),
            );
        }
        let g = blob.len() / 3;
        if g == 0 || g > MAX_GATES {
            continue;
        }
        let entry = lengths.entry(g).or_insert_with(|| (0, Vec::new()));
        entry.0 += 1;
        if entry.1.len() < member_cap {
            entry.1.push(blob.to_vec());
        } else {
            let j = rng.random_range(0..entry.0) as usize;
            if j < member_cap {
                entry.1[j] = blob.to_vec();
            }
        }
    }
    flush(current.take(), &mut lengths, &mut rng)?;
    let n = counters.done_prefixes.fetch_add(1, Ordering::Relaxed) + 1;
    if n % 16 == 0 || n == 256 {
        let emitted: Vec<String> = MODE_LABELS
            .iter()
            .zip(counters.emitted.iter())
            .filter(|(_, e)| e.load(Ordering::Relaxed) > 0)
            .map(|(l, e)| format!("{l}={}", e.load(Ordering::Relaxed)))
            .collect();
        eprintln!(
            "[build] prefixes={n}/256 pools={} pairs={} emitted {} oracle-calls={} ({:.0}s)",
            counters.pools.load(Ordering::Relaxed),
            counters.pairs_tested.load(Ordering::Relaxed),
            emitted.join(" "),
            oracle.calls.load(Ordering::Relaxed),
            started.elapsed().as_secs_f64()
        );
    }
    Ok(())
}

fn main() {
    if let Err(error) = run() {
        eprintln!("true_friend_probe: {error}");
        std::process::exit(1);
    }
}

#[allow(clippy::too_many_lines)]
fn run() -> AnyResult<()> {
    let args: Vec<String> = std::env::args().collect();
    if args.len() < 3 {
        eprintln!(
            "usage: true_friend_probe POOLS_COMPOSITE FROZEN_REGULAR_DIR [--out DIR] \
             [--s1 N] [--s2 N] [--s3 N] [--s4 N] [--mega-per-len N] [--seed N] \
             [--self-test] [--no-mega] [--mega KEY_HEX]..."
        );
        std::process::exit(2);
    }
    let pools_path = &args[1];
    let frozen_dir = &args[2];
    let flag = |name: &str| {
        args.iter()
            .position(|a| a == name)
            .and_then(|i| args.get(i + 1))
    };
    let num =
        |name: &str, default: usize| flag(name).and_then(|s| s.parse().ok()).unwrap_or(default);
    let out_dir = PathBuf::from(flag("--out").map_or("true_friend_probe_out", |s| s.as_str()));
    let s1_n = num("--s1", 3000);
    let s2_n = num("--s2", 1500);
    let s3_n = num("--s3", 1500);
    let s4_n = num("--s4", 300);
    let mega_per_len = num("--mega-per-len", 16);
    let seed = num("--seed", 0x7f) as u64;
    let parse_key = |hexstr: &str| -> Option<[u8; 16]> {
        if hexstr.len() != 32 {
            return None;
        }
        let mut k = [0u8; 16];
        for (b, chunk) in k.iter_mut().zip(hexstr.as_bytes().chunks(2)) {
            *b = u8::from_str_radix(std::str::from_utf8(chunk).ok()?, 16).ok()?;
        }
        Some(k)
    };
    let mut mega_keys: Vec<[u8; 16]> = Vec::new();
    for (i, a) in args.iter().enumerate() {
        if a == "--mega" {
            if let Some(k) = args.get(i + 1).and_then(|s| parse_key(s)) {
                mega_keys.push(k);
            }
        }
    }
    if mega_keys.is_empty() && !args.iter().any(|a| a == "--no-mega") {
        for hexstr in [
            "66ca88066ac7c7a878442082635feaeb",
            "94988b9e2030eb36b4e9e428cde51ab6",
            "0ff4390c610096c8835461fe3ba26ce2",
            "76e8d554f5e45f9d82147be436895466",
        ] {
            mega_keys.push(parse_key(hexstr).unwrap());
        }
    }

    let mut options = Options::default();
    options.create_if_missing(false);
    let pools = DB::open_for_read_only(&options, pools_path, false)?;
    if pools.get(COMPOSITE_FORMAT_MARKER)?.as_deref() != Some(b"1") {
        return Err(format!("{pools_path} is not a curated-full composite-v1 store").into());
    }
    let oracle = Oracle::new(FrozenDb::open(frozen_dir, None));
    std::fs::create_dir_all(&out_dir)?;
    let started = std::time::Instant::now();

    if args.iter().any(|a| a == "--self-test") {
        let show = |label: &str, gates: &[[u16; 3]]| {
            eprintln!("[self-test] {label}: {:?}", oracle.minlen(gates));
        };
        show("single gate", &[[0, 1, 2]]);
        show("gate twice (identity)", &[[0, 1, 2], [0, 1, 2]]);
        show("two disjoint gates", &[[0, 1, 2], [3, 4, 5]]);
        show("two overlapping gates", &[[0, 1, 2], [1, 0, 2]]);
        show("three-gate word", &[[0, 1, 2], [3, 0, 4], [2, 3, 1]]);
        let mut rng = rand::rngs::StdRng::seed_from_u64(seed);
        for key in &mega_keys {
            let sampled = sample_mega(&pools, key, &(1..=11).collect::<Vec<_>>(), 1, &mut rng);
            if let Some((g, members)) = sampled.iter().next() {
                eprintln!(
                    "[self-test] mega {} shortest sampled ({g} gates): {:?}",
                    hex(key),
                    oracle.minlen(&members[0])
                );
            }
        }
        return Ok(());
    }

    if args.iter().any(|a| a == "--build") {
        let member_cap = num("--member-cap", 48);
        let pair_cap = num("--pair-cap", 2500);
        let mut outs: [Option<Arc<DB>>; MODE_COUNT] = Default::default();
        for (mode, name) in MODE_FLAGS.iter().enumerate() {
            if let Some(path) = flag(name) {
                outs[mode] = Some(open_fresh_composite(path)?);
            }
        }
        if outs.iter().all(Option::is_none) {
            return Err(
                format!("--build requires at least one of {}", MODE_FLAGS.join(" ")).into(),
            );
        }
        let counters = BuildCounters::new();
        (0..=255u8).into_par_iter().try_for_each(|p| {
            build_prefix(
                &pools, &oracle, &outs, p, member_cap, pair_cap, seed, &counters, &started,
            )
        })?;
        eprintln!(
            "[build] scan done: pools={} (identity skipped {}) members seen={} sampled={} pairs tested={} ({:.0}s)",
            counters.pools.load(Ordering::Relaxed),
            counters.pools_identity.load(Ordering::Relaxed),
            counters.members_seen.load(Ordering::Relaxed),
            counters.members_sampled.load(Ordering::Relaxed),
            counters.pairs_tested.load(Ordering::Relaxed),
            started.elapsed().as_secs_f64()
        );
        for (mode, db) in outs.iter().enumerate() {
            if let Some(db) = db {
                finalize_output(db, MODE_LABELS[mode])?;
            }
        }
        eprintln!(
            "[build] oracle: calls={} store-lookups={} table-contradictions={} total {:.0}s",
            oracle.calls.load(Ordering::Relaxed),
            oracle.store_lookups.load(Ordering::Relaxed),
            oracle.table_contradictions.load(Ordering::Relaxed),
            started.elapsed().as_secs_f64()
        );
        return Ok(());
    }

    // ---- Sampling pass.
    let mut rng = rand::rngs::StdRng::seed_from_u64(seed);
    let mut chosen: FxHashSet<[u8; 16]> = FxHashSet::default();
    let (mut s1, mut s2, mut s3, mut s4) = (Vec::new(), Vec::new(), Vec::new(), Vec::new());
    let mut attempts = 0usize;
    let budget = (s1_n + s2_n + s3_n + s4_n) * 400;
    while (s1.len() < s1_n || s2.len() < s2_n || s3.len() < s3_n || s4.len() < s4_n)
        && attempts < budget
    {
        attempts += 1;
        let mut probe = [0u8; 16];
        rng.fill(&mut probe);
        let Some(pool) = read_pool(&pools, &probe, 4097) else {
            continue;
        };
        if pool.truncated || !chosen.insert(pool.key) {
            continue;
        }
        let size = pool.members.len();
        let cohort = |g: usize| pool.members.iter().filter(|m| m.len() == g).count();
        if size == 2 && s1.len() < s1_n {
            s1.push(pool);
        } else if (3..=64).contains(&size) && cohort(6) >= 2 && s2.len() < s2_n {
            s2.push(pool);
        } else if (3..=4096).contains(&size) && cohort(7) >= 2 && s3.len() < s3_n {
            s3.push(pool);
        } else if (65..=4096).contains(&size) && s4.len() < s4_n {
            s4.push(pool);
        }
    }
    eprintln!(
        "[sample] attempts={attempts} s1={} s2={} s3={} s4={} ({:.1}s)",
        s1.len(),
        s2.len(),
        s3.len(),
        s4.len(),
        started.elapsed().as_secs_f64()
    );

    let process = |pools_in: Vec<Pool>,
                   label: &str,
                   restrict_len: Option<usize>,
                   member_cap: usize,
                   pair_cap: usize|
     -> StratumStats {
        let done = AtomicU64::new(0);
        let total = pools_in.len() as u64;
        pools_in
            .into_par_iter()
            .map(|pool| {
                let mut stats = StratumStats::default();
                let mut rng = rand::rngs::StdRng::seed_from_u64(
                    seed ^ u64::from_le_bytes(pool.key[..8].try_into().unwrap()),
                );
                let width = semantic_width(&pool.members);
                let Some(delta) = pool_delta(&oracle, &pool.members) else {
                    return stats;
                };
                if delta == 0 {
                    return stats;
                }
                let mut members: Vec<Vec<[u16; 3]>> = match restrict_len {
                    Some(g) => pool
                        .members
                        .iter()
                        .filter(|m| m.len() == g)
                        .cloned()
                        .collect(),
                    None => pool.members.clone(),
                };
                if members.len() > member_cap {
                    for i in 0..member_cap {
                        let j = rng.random_range(i..members.len());
                        members.swap(i, j);
                    }
                    members.truncate(member_cap);
                }
                stats.pools = 1;
                let degrees = test_pool_pairs(
                    &members, delta, width, &oracle, &mut stats, pair_cap, &mut rng,
                );
                if let Some(g) = restrict_len {
                    let cohort = members.len();
                    let tight = stats.circuits_tight as usize;
                    let a: u64 = stats.cells.values().map(|c| c.a_pass).sum();
                    let c: u64 = stats.cells.values().map(|c| c.c_pass).sum();
                    let b: u64 = stats.cells.values().map(|c| c.b_pass).sum();
                    let bp: u64 = stats.cells.values().map(|c| c.bp_pass).sum();
                    stats
                        .degree_rows
                        .push((hex(&pool.key), g, cohort, tight, a, c, b, bp));
                    let _ = degrees;
                }
                let n = done.fetch_add(1, Ordering::Relaxed) + 1;
                if n % 1000 == 0 || n == total {
                    eprintln!(
                        "[{label}] {n}/{total} oracle: calls={} store={} ({:.1}s)",
                        oracle.calls.load(Ordering::Relaxed),
                        oracle.store_lookups.load(Ordering::Relaxed),
                        started.elapsed().as_secs_f64()
                    );
                }
                stats
            })
            .reduce(StratumStats::default, StratumStats::merge)
    };

    let s1_stats = process(s1, "S1 size-2", None, 2, 1);
    let s2_stats = process(s2, "S2 L6-cohort", Some(6), 12, 66);
    let s3_stats = process(s3, "S3 L7-cohort", Some(7), 12, 66);
    let s4_stats = process(s4, "S4 mid-pool", None, 12, 66);

    // ---- S5.
    let mut s5_stats = StratumStats::default();
    let mut s5_rows: Vec<String> = Vec::new();
    for key in &mega_keys {
        let lengths: Vec<usize> = (1..=11).collect();
        let sampled = sample_mega(&pools, key, &lengths, mega_per_len, &mut rng);
        let all: Vec<&Vec<[u16; 3]>> = sampled.values().flatten().collect();
        if all.is_empty() {
            eprintln!("[S5] key {} not found; skipped", hex(key));
            continue;
        }
        let shortest = all
            .iter()
            .min_by_key(|m| m.len())
            .copied()
            .cloned()
            .unwrap();
        let width = shortest
            .iter()
            .flatten()
            .copied()
            .max()
            .map_or(0, |w| w + 1);
        let delta = match oracle.minlen(&shortest) {
            MinLen::Exact(d) => d as usize,
            MinLen::AtLeast4 { witness } => (witness as usize).min(shortest.len()),
        };
        if delta == 0 {
            eprintln!("[S5] key {} is the identity pool; skipped", hex(key));
            continue;
        }
        for (&g, members) in &sampled {
            if members.len() < 2 {
                continue;
            }
            let mut stats = StratumStats::default();
            stats.pools = 1;
            let mut prng = rand::rngs::StdRng::seed_from_u64(seed ^ g as u64);
            let _ = test_pool_pairs(members, delta, width, &oracle, &mut stats, 120, &mut prng);
            let a: u64 = stats.cells.values().map(|c| c.a_pass).sum();
            let c: u64 = stats.cells.values().map(|c| c.c_pass).sum();
            let b: u64 = stats.cells.values().map(|c| c.b_pass).sum();
            let bp: u64 = stats.cells.values().map(|c| c.bp_pass).sum();
            s5_rows.push(format!(
                "{},{},{},{},{},{},{},{},{}",
                hex(key),
                delta,
                g,
                members.len(),
                stats.circuits_tight,
                a,
                c,
                b,
                bp
            ));
            eprintln!(
                "[S5] key={} delta={delta} gates={g} sampled={} tight={} a={a} c={c} b_ordered={b} bp_ordered={bp}",
                hex(key),
                members.len(),
                stats.circuits_tight
            );
            s5_stats = s5_stats.merge(stats);
        }
    }

    // ---- Report.
    let mut summary = String::new();
    let mut cells_csv: Vec<String> = Vec::new();
    for (label, st) in [
        ("S1 size-2 pools", &s1_stats),
        ("S2 6-gate cohorts", &s2_stats),
        ("S3 7-gate cohorts", &s3_stats),
        ("S4 mid pools", &s4_stats),
        ("S5 mega pools", &s5_stats),
    ] {
        let tested: u64 = st.cells.values().map(|c| c.tested).sum();
        let a: u64 = st.cells.values().map(|c| c.a_pass).sum();
        let c: u64 = st.cells.values().map(|c| c.c_pass).sum();
        let b: u64 = st.cells.values().map(|c| c.b_pass).sum();
        let bp: u64 = st.cells.values().map(|c| c.bp_pass).sum();
        summary.push_str(&format!(
            "\n== {label}: pools={} circuits={} tight={} ({:.1}%) unordered-pairs={} | (a) {} ({:.2}%) | (c) ordered {} | (b) ordered {} ({:.2}% of 2x) | (b') {}\n",
            st.pools,
            st.circuits,
            st.circuits_tight,
            st.circuits_tight as f64 * 100.0 / (st.circuits.max(1)) as f64,
            tested,
            a,
            a as f64 * 100.0 / tested.max(1) as f64,
            c,
            b,
            b as f64 * 100.0 / (2 * tested.max(1)) as f64,
            bp,
        ));
        summary.push_str(&format!(
            "   keys with >=1 (b)-friend: tight {}/{}  junky {}/{}   (b)-pass leeway used 0/1/2: {}/{}/{}\n",
            st.keys_served_tight,
            st.keys_total_tight,
            st.keys_served_junky,
            st.keys_total_junky,
            st.b_need_hist[0],
            st.b_need_hist[1],
            st.b_need_hist[2],
        ));
        summary.push_str(&format!(
            "{:>11} {:>9} {:>8} {:>10} {:>10} {:>10}\n",
            "(lenA,lenB)", "tested", "(a)", "(c) ord", "(b) ord", "(b') ord"
        ));
        for (&(x, y), cell) in &st.cells {
            summary.push_str(&format!(
                "{:>11} {:>9} {:>8} {:>10} {:>10} {:>10}\n",
                format!("({x},{y})"),
                cell.tested,
                cell.a_pass,
                cell.c_pass,
                cell.b_pass,
                cell.bp_pass,
            ));
            cells_csv.push(format!(
                "{label},{x},{y},{},{},{},{},{}",
                cell.tested, cell.a_pass, cell.c_pass, cell.b_pass, cell.bp_pass
            ));
        }
    }
    summary.push_str(&format!(
        "\noracle: calls={} store-lookups={} table-contradictions={} elapsed={:.1}s\n",
        oracle.calls.load(Ordering::Relaxed),
        oracle.store_lookups.load(Ordering::Relaxed),
        oracle.table_contradictions.load(Ordering::Relaxed),
        started.elapsed().as_secs_f64()
    ));
    print!("{summary}");

    std::fs::File::create(out_dir.join("summary.txt"))?.write_all(summary.as_bytes())?;
    let mut f = std::fs::File::create(out_dir.join("pair_cells.csv"))?;
    writeln!(
        f,
        "stratum,len_a,len_b,tested,a_pass,c_pass_ordered,b_pass_ordered,bp_pass_ordered"
    )?;
    for row in cells_csv {
        writeln!(f, "{row}")?;
    }
    let mut f = std::fs::File::create(out_dir.join("per_key_degree.csv"))?;
    writeln!(
        f,
        "key,gates,cohort,tight,a_pairs,c_ordered,b_ordered,bp_ordered"
    )?;
    for st in [&s2_stats, &s3_stats] {
        for (key, g, cohort, tight, a, c, b, bp) in &st.degree_rows {
            writeln!(f, "{key},{g},{cohort},{tight},{a},{c},{b},{bp}")?;
        }
    }
    let mut f = std::fs::File::create(out_dir.join("mega_cohorts.csv"))?;
    writeln!(
        f,
        "key,delta,gates,sampled,tight,a_pairs,c_ordered,b_ordered,bp_ordered"
    )?;
    for row in s5_rows {
        writeln!(f, "{row}")?;
    }
    eprintln!("outputs in {}", out_dir.display());
    Ok(())
}
