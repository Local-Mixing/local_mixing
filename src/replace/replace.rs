// Replacement code used in the mixing methods

use crate::replace::frozen::FrozenDb;
use crate::replace::mixing::split_into_random_chunk_ranges;
use crate::{
    circuit::circuit::{CircuitSeq, Permutation},
    random::random_data::{
        contiguous_convex, find_convex_subcircuit_max_gates, find_convex_subcircuit_max_wires,
        simple_find_convex_subcircuit,
    },
};
use rand::Rng;
use rayon::iter::IndexedParallelIterator;
use rayon::iter::IntoParallelIterator;
use rayon::iter::ParallelIterator;
use std::fs::File;
use std::io::{BufWriter, Write};

use dashmap::DashMap;
use once_cell::sync::Lazy;
use std::sync::atomic::{AtomicBool, AtomicU32, AtomicU64, AtomicUsize, Ordering};
use std::sync::{Mutex, OnceLock};
use std::{
    cmp::{max, min},
    time::Instant,
};
// use rand::prelude::IndexedRandom;

// Global histogram: (before_gates, after_gates) -> count, accumulated across all rounds
pub static COMPRESSION_HISTOGRAM: Lazy<DashMap<(u8, u8), u64>> = Lazy::new(DashMap::new);

// Attempt-level compression lookup stats, enabled with COMPRESS_ATTEMPT_STATS=1.
// Keyed by (window_gates, window_used_wires): every canonicalized DB probe, the
// probes whose key was present, and the hits offering a shorter candidate.
// Used to decide which DB-build axis (more gates at high min_n vs denser
// low-min_n coverage) actually starves compression.
pub static ATTEMPT_HISTOGRAM: Lazy<DashMap<(u8, u8), u64>> = Lazy::new(DashMap::new);
pub static ATTEMPT_HIT_HISTOGRAM: Lazy<DashMap<(u8, u8), u64>> = Lazy::new(DashMap::new);
pub static ATTEMPT_SHORTER_HISTOGRAM: Lazy<DashMap<(u8, u8), u64>> = Lazy::new(DashMap::new);

fn attempt_stats_enabled() -> bool {
    static ON: OnceLock<bool> = OnceLock::new();
    *ON.get_or_init(|| std::env::var("COMPRESS_ATTEMPT_STATS").is_ok())
}

/// Dump the attempt-level lookup stats as CSV lines (g,w,attempts,hits,shorter).
pub fn print_attempt_stats() {
    if ATTEMPT_HISTOGRAM.is_empty() {
        return;
    }
    let mut keys: Vec<(u8, u8)> = ATTEMPT_HISTOGRAM.iter().map(|e| *e.key()).collect();
    keys.sort_unstable();
    println!("attempt_stats_csv g,w,attempts,hits,shorter");
    for k in keys {
        let attempts = ATTEMPT_HISTOGRAM.get(&k).map_or(0, |v| *v);
        let hits = ATTEMPT_HIT_HISTOGRAM.get(&k).map_or(0, |v| *v);
        let shorter = ATTEMPT_SHORTER_HISTOGRAM.get(&k).map_or(0, |v| *v);
        println!(
            "attempt_stats_csv {},{},{},{},{}",
            k.0, k.1, attempts, hits, shorter
        );
    }
}

// Global histograms for EXPANSIONS made in the shuffle-shoot-shuffle game, accumulated
// across all rounds. One is keyed by gate counts, the other by distinct-wire counts.
pub static EXPANSION_HISTOGRAM: Lazy<DashMap<(u8, u8), u64>> = Lazy::new(DashMap::new);
pub static EXPANSION_WIRE_HISTOGRAM: Lazy<DashMap<(u8, u8), u64>> = Lazy::new(DashMap::new);

pub static RECORD_ENABLED: AtomicBool = AtomicBool::new(false);
pub static REC_ROUND: AtomicUsize = AtomicUsize::new(0);
pub static REC_PASS: AtomicUsize = AtomicUsize::new(0);
pub static REC_ITER: AtomicUsize = AtomicUsize::new(0);
static REC_SEQ: AtomicUsize = AtomicUsize::new(0);
static REC_SINK: Lazy<Mutex<Option<BufWriter<File>>>> = Lazy::new(|| Mutex::new(None));

pub const TAG_NEW: u32 = u32::MAX;
pub static TRACK_SURVIVORS: AtomicBool = AtomicBool::new(false);
pub static GEN_MODE: AtomicBool = AtomicBool::new(false);
pub static OUTGOING_GEN_MODE: AtomicBool = AtomicBool::new(false);
pub static FORCED_COLLISIONS: AtomicUsize = AtomicUsize::new(0);
pub static MAX_FANOUT: AtomicUsize = AtomicUsize::new(50);
pub static MIN_MEDIAN_LEEWAY: AtomicUsize = AtomicUsize::new(10);
pub static SAMF_TARGET: AtomicUsize = AtomicUsize::new(0);
pub static INCOMING_RANK_MODE: AtomicUsize = AtomicUsize::new(IncomingRankMode::Random as usize);

#[repr(usize)]
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum IncomingRankMode {
    Random = 0,
    Fanout = 1,
}

// Running per-wire touch counts of picked replacements, used to steer ancilla assignment
// toward globally cold wires (ANCILLA_BALANCE=1). Uneven wire usage is what caps leeway
// below matched-random levels and lets hot wires accumulate outlier fanout, so ancillas —
// the one wire choice the transform is free to make — should preferentially reuse the
// least-touched wires. Sized for the full u16 wire space; relative order is all that matters.
pub static WIRE_LOAD: Lazy<Vec<AtomicU32>> =
    Lazy::new(|| (0..=u16::MAX as usize).map(|_| AtomicU32::new(0)).collect());

pub fn ancilla_balance_enabled() -> bool {
    static ENABLED: OnceLock<bool> = OnceLock::new();
    *ENABLED.get_or_init(|| {
        std::env::var("ANCILLA_BALANCE")
            .map(|value| {
                let value = value.trim();
                !value.is_empty()
                    && !matches!(
                        value.to_ascii_lowercase().as_str(),
                        "0" | "false" | "off" | "no"
                    )
            })
            .unwrap_or(false)
    })
}

/// Record the wires touched by a picked replacement into the global load table.
pub fn note_wire_use(gates: &[[u16; 3]]) {
    if !ancilla_balance_enabled() {
        return;
    }
    for gate in gates {
        for &wire in gate {
            WIRE_LOAD[wire as usize].fetch_add(1, Ordering::Relaxed);
        }
    }
}

// Coldest-first ordering; a stable sort after shuffling keeps the shuffled order as the
// random tiebreak among equally loaded wires.
fn order_coldest_first(available: &mut [u16], load: impl Fn(u16) -> u32) {
    available.sort_by_key(|&wire| load(wire));
}

pub fn shuffled_unused_wires(n: usize, used_wires: &[u16], rng: &mut impl Rng) -> Vec<u16> {
    let mut used_mask = vec![false; n];
    for &wire in used_wires {
        if let Some(slot) = used_mask.get_mut(wire as usize) {
            *slot = true;
        }
    }
    let mut available: Vec<u16> = (0..n as u16)
        .filter(|&wire| !used_mask[wire as usize])
        .collect();
    rand::seq::SliceRandom::shuffle(available.as_mut_slice(), rng);
    if ancilla_balance_enabled() {
        order_coldest_first(&mut available, |wire| {
            WIRE_LOAD[wire as usize].load(Ordering::Relaxed)
        });
    }
    available
}

// Map a stored candidate from canonical wire space into circuit wire space: undo the
// canonical direction, apply the canonicalization order, then relabel onto the window's
// used wires (drawing random unused wires for any extra ancillas). Candidates must be in
// circuit space BEFORE context-aware ranking — their canonical wire labels are meaningless
// next to the surrounding circuit gates.
fn candidate_to_circuit_space(
    mut repl: CircuitSeq,
    is_reversed: bool,
    final_order: &Permutation,
    used: &[u16],
    n: usize,
    rng: &mut impl Rng,
) -> CircuitSeq {
    if is_reversed {
        repl.gates.reverse();
    }
    let repl_n = repl.max_wire() + 1;
    let mut order_data = final_order.data.clone();
    while order_data.len() < repl_n {
        let i = order_data.len();
        order_data.push(i);
    }
    repl.rewire(
        &Permutation { data: order_data },
        std::cmp::max(repl_n, final_order.data.len()),
    );

    let repl_n_b = repl.max_wire() + 1;
    let mut used_ext = used.to_vec();
    if used_ext.len() < repl_n_b {
        let available = shuffled_unused_wires(n, &used_ext, rng);
        let mut avail = available.into_iter();
        while used_ext.len() < repl_n_b {
            match avail.next() {
                Some(w) => used_ext.push(w),
                None => break,
            }
        }
    }
    CircuitSeq::unrewire_subcircuit(&repl, &used_ext)
}

// Record one expansion: `before`/`after` gate counts and `before_wires`/`after_wires`
// distinct-wire counts.
pub fn record_expansion(before: usize, after: usize, before_wires: usize, after_wires: usize) {
    *EXPANSION_HISTOGRAM
        .entry((before as u8, after as u8))
        .or_insert(0) += 1;
    *EXPANSION_WIRE_HISTOGRAM
        .entry((before_wires as u8, after_wires as u8))
        .or_insert(0) += 1;
}

#[inline]
pub fn record_enabled() -> bool {
    RECORD_ENABLED.load(Ordering::Relaxed)
}

pub fn record_init(path: &str) {
    let f = File::create(path).expect("Failed to create replacement record file");
    let mut w = BufWriter::new(f);
    writeln!(
        w,
        "# per-replacement log. out=start-end are momentary working-buffer gate indices\n\
         # seq stage round ctx out_start-out_end out_gates in_gates wires"
    )
    .ok();
    *REC_SINK.lock().unwrap() = Some(w);
    RECORD_ENABLED.store(true, Ordering::Relaxed);
}

pub fn record_replacement(
    stage: &str,
    ctx: usize,
    out_start: usize,
    out_end: usize,
    in_gates: usize,
    wires: &[u16],
) {
    if !record_enabled() {
        return;
    }
    let seq = REC_SEQ.fetch_add(1, Ordering::Relaxed);
    let round = REC_ROUND.load(Ordering::Relaxed);
    let mut ws = wires.to_vec();
    ws.sort_unstable();
    ws.dedup();
    let wires_str = ws
        .iter()
        .map(|w| w.to_string())
        .collect::<Vec<_>>()
        .join(",");
    let last = out_end.saturating_sub(1);
    if let Some(w) = REC_SINK.lock().unwrap().as_mut() {
        writeln!(
            w,
            "seq={} stage={} round={} ctx={} out={}-{} out_gates={} in_gates={} wires=[{}]",
            seq,
            stage,
            round,
            ctx,
            out_start,
            last,
            out_end.saturating_sub(out_start),
            in_gates,
            wires_str
        )
        .ok();
    }
}

pub fn record_finish() {
    if let Some(w) = REC_SINK.lock().unwrap().as_mut() {
        w.flush().ok();
    }
}

#[inline]
pub fn track_survivors() -> bool {
    TRACK_SURVIVORS.load(Ordering::Relaxed)
}

#[inline]
pub fn gen_mode() -> bool {
    GEN_MODE.load(Ordering::Relaxed)
}

#[inline]
pub fn outgoing_gen_mode() -> bool {
    OUTGOING_GEN_MODE.load(Ordering::Relaxed)
}

#[inline]
pub fn incoming_rank_mode() -> IncomingRankMode {
    match INCOMING_RANK_MODE.load(Ordering::Relaxed) {
        1 => IncomingRankMode::Fanout,
        _ => IncomingRankMode::Random,
    }
}

pub fn median_floor(values: &[u32]) -> u32 {
    if values.is_empty() {
        return 0;
    }
    let mut sorted = values.to_vec();
    sorted.sort_unstable();
    let n = sorted.len();
    if n % 2 == 1 {
        sorted[n / 2]
    } else {
        ((sorted[n / 2 - 1] as u64 + sorted[n / 2] as u64) / 2) as u32
    }
}

#[inline]
pub fn new_gate_tag(window_tags: &[u32]) -> u32 {
    if gen_mode() {
        median_floor(window_tags).saturating_add(1)
    } else {
        TAG_NEW
    }
}

pub const FANOUT_TARGET: [f64; 5] = [0.25, 0.40, 0.20, 0.10, 0.05];

fn fanout_bucket(f: usize) -> usize {
    f.min(4)
}

pub fn gate_fanouts(gates: &[[u16; 3]]) -> Vec<usize> {
    let n = gates.len();
    let mut out = vec![0usize; n];
    for i in 0..n {
        let w = gates[i][0];
        for g in &gates[i + 1..] {
            if g[0] == w {
                break;
            }
            if g[1] == w || g[2] == w {
                out[i] += 1;
            }
        }
    }
    out
}

pub fn gate_leeways(gates: &[[u16; 3]]) -> Vec<usize> {
    use crate::circuit::circuit::Gate;
    let n = gates.len();
    let mut out = vec![0usize; n];
    for i in 0..n {
        let mut l = 0usize;
        let mut j = i;
        while j > 0 && !Gate::collides_index(&gates[j - 1], &gates[i]) {
            l += 1;
            j -= 1;
        }
        let mut r = 0usize;
        let mut k = i;
        while k + 1 < n && !Gate::collides_index(&gates[k + 1], &gates[i]) {
            r += 1;
            k += 1;
        }
        out[i] = l + r;
    }
    out
}

pub fn cand_features(
    window: &[[u16; 3]],
    left: &[[u16; 3]],
    right: &[[u16; 3]],
) -> crate::replace::ranking::CandFeatures {
    use crate::circuit::circuit::Gate;
    use crate::replace::ranking::CandFeatures;

    let threshold = MIN_MEDIAN_LEEWAY.load(Ordering::Relaxed);
    let mut wires = std::collections::HashSet::new();
    for g in window {
        wires.insert(g[0]);
        wires.insert(g[1]);
        wires.insert(g[2]);
    }

    let mut buckets = [0usize; 5];
    let mut zero_fanout = 0usize;
    let mut low_leeway = 0usize;
    let mut max_fanout = 0usize;
    let mut leeways: Vec<usize> = Vec::with_capacity(window.len());
    let mut touch_counts: std::collections::HashMap<u16, usize> = std::collections::HashMap::new();
    for gate in window {
        for &wire in gate {
            *touch_counts.entry(wire).or_insert(0) += 1;
        }
    }
    for i in 0..window.len() {
        let target = window[i][0];
        let mut fanout = 0usize;
        let mut rewritten = false;
        for g in &window[i + 1..] {
            if g[0] == target {
                rewritten = true;
                break;
            }
            if g[1] == target || g[2] == target {
                fanout += 1;
            }
        }
        if !rewritten {
            for g in right {
                if g[0] == target {
                    break;
                }
                if g[1] == target || g[2] == target {
                    fanout += 1;
                }
            }
        }

        let cur = &window[i];
        let mut leeway = 0usize;
        let mut blocked = false;
        for g in window[..i].iter().rev() {
            if Gate::collides_index(g, cur) {
                blocked = true;
                break;
            }
            leeway += 1;
        }
        if !blocked {
            for g in left.iter().rev() {
                if Gate::collides_index(g, cur) {
                    break;
                }
                leeway += 1;
            }
        }
        blocked = false;
        for g in &window[i + 1..] {
            if Gate::collides_index(g, cur) {
                blocked = true;
                break;
            }
            leeway += 1;
        }
        if !blocked {
            for g in right {
                if Gate::collides_index(g, cur) {
                    break;
                }
                leeway += 1;
            }
        }

        buckets[fanout_bucket(fanout)] += 1;
        zero_fanout += usize::from(fanout == 0);
        low_leeway += usize::from(leeway < threshold);
        max_fanout = max_fanout.max(fanout);
        leeways.push(leeway);
    }

    leeways.sort_unstable();
    let median_leeway = leeways.get(leeways.len() / 2).copied().unwrap_or(0);
    let max_wire_touch = touch_counts.values().copied().max().unwrap_or(0);

    CandFeatures {
        size: window.len(),
        wires_spanned: wires.len(),
        low_leeway_count: low_leeway,
        zero_fanout_count: zero_fanout,
        fanout_buckets: buckets,
        max_fanout,
        median_leeway,
        max_wire_touch,
    }
}

fn fanout_pick_index(candidates: &[CircuitSeq], left: &[[u16; 3]], right: &[[u16; 3]]) -> usize {
    let features: Vec<_> = candidates
        .iter()
        .map(|candidate| cand_features(&candidate.gates, left, right))
        .collect();
    crate::replace::ranking::incoming()
        .order(&features)
        .first()
        .copied()
        .unwrap_or(0)
}

// Write a (before, after) -> count histogram to `csv_path` and a human-readable log to
// `log_path`. `unit` labels the rows in the log (e.g. "gates" or "wires").
fn write_before_after_histogram(
    hist: &DashMap<(u8, u8), u64>,
    csv_path: &str,
    log_path: &str,
    unit: &str,
) {
    let mut entries: Vec<((u8, u8), u64)> = hist.iter().map(|e| (*e.key(), *e.value())).collect();
    entries.sort_by_key(|&((before, after), _)| (before, after));
    let mut f = File::create(csv_path).expect("Failed to create histogram CSV");
    writeln!(f, "before,after,count").expect("write");
    for ((before, after), count) in &entries {
        writeln!(f, "{},{},{}", before, after, count).expect("write");
    }
    println!("Histogram written to {}", csv_path);

    let mut log = File::create(log_path).expect("Failed to create histogram log");
    let before_vals: Vec<u8> = {
        let mut v: Vec<u8> = entries.iter().map(|&((b, _), _)| b).collect();
        v.dedup();
        v
    };
    for before in before_vals {
        let group: Vec<_> = entries.iter().filter(|&((b, _), _)| *b == before).collect();
        let total: u64 = group.iter().map(|(_, c)| c).sum();
        writeln!(log, "{} {} before (total: {}):", before, unit, total).expect("write");
        for ((_, after), count) in &group {
            writeln!(log, "  -> {} {}: {}", after, unit, count).expect("write");
        }
    }
    println!("Log written to {}", log_path);
}

pub fn write_expansion_histogram(path: &str) {
    write_before_after_histogram(&EXPANSION_HISTOGRAM, path, "expansion_log.txt", "gates");
}

pub fn write_expansion_wire_histogram(path: &str) {
    write_before_after_histogram(
        &EXPANSION_WIRE_HISTOGRAM,
        path,
        "expansion_wire_log.txt",
        "wires",
    );
}

pub fn write_compression_histogram(path: &str) {
    let mut entries: Vec<((u8, u8), u64)> = COMPRESSION_HISTOGRAM
        .iter()
        .map(|e| (*e.key(), *e.value()))
        .collect();
    entries.sort_by_key(|&((before, after), _)| (before, after));
    let mut f = File::create(path).expect("Failed to create histogram CSV");
    writeln!(f, "before,after,count").expect("write");
    for ((before, after), count) in &entries {
        writeln!(f, "{},{},{}", before, after, count).expect("write");
    }
    println!("Compression histogram written to {}", path);

    let log_path = "compression_log.txt";
    let mut log = File::create(log_path).expect("Failed to create compression log");
    let before_vals: Vec<u8> = {
        let mut v: Vec<u8> = entries.iter().map(|&((b, _), _)| b).collect();
        v.dedup();
        v
    };
    for before in before_vals {
        let group: Vec<_> = entries.iter().filter(|&((b, _), _)| *b == before).collect();
        let total: u64 = group.iter().map(|(_, c)| c).sum();
        writeln!(log, "{} gates before (total: {}):", before, total).expect("write");
        for ((_, after), count) in &group {
            writeln!(log, "  -> {} gates: {}", after, count).expect("write");
        }
    }
    println!("Compression log written to {}", log_path);
    println!("Written to log");
}

// Return a random contiguous subcircuit, its starting index (gate), and ending index
pub fn random_subcircuit(circuit: &CircuitSeq) -> (CircuitSeq, usize, usize) {
    let len = circuit.gates.len();

    if circuit.gates.len() == 0 {
        return (CircuitSeq { gates: Vec::new() }, 0, 0);
    }

    let mut rng = rand::rng();
    //get size with more bias to lower length subcircuits
    let a = rng.random_range(0..len);

    // pick one of 1, 2, 4, 8
    let shift = rng.random_range(0..4);
    let upper = 1 << shift;

    let mut b = (a + (6 + rng.random_range(0..upper))) as usize;

    if b > len {
        b = len;
    }

    if a == b {
        if b < len - 1 {
            b += 1;
        } else {
            b -= 1;
        }
    }

    let start = min(a, b);
    let end = max(a, b);

    let subcircuit = circuit.gates[start..end].to_vec();

    (CircuitSeq { gates: subcircuit }, start, end)
}

pub fn random_subcircuit_max(circuit: &CircuitSeq, max_len: usize) -> (CircuitSeq, usize, usize) {
    let len = circuit.gates.len();
    if len == 0 {
        return (CircuitSeq { gates: Vec::new() }, 0, 0);
    }

    let mut rng = rand::rng();

    let start = rng.random_range(0..len);

    let remaining = len - start;
    let allowed_len = remaining.min(max_len);

    let shift = rng.random_range(0..4); // 0..3
    let mut sub_len = 1 << shift; // 1,2,4,8
    if sub_len > allowed_len {
        sub_len = allowed_len;
    }

    sub_len = sub_len.max(1);

    let end = start + sub_len;
    let subcircuit = circuit.gates[start..end].to_vec();

    (CircuitSeq { gates: subcircuit }, start, end)
}

// ---------------------------------------------------------------------------
// Frozen-store lookup cache.
//
// The frozen database is immutable for the lifetime of the
// process, so a lookup's result can never change: caching (key -> value bytes,
// including "absent") is exact, not approximate. Windows drawn by the
// expansion/compression games repeat heavily (SAMF templates recur all over
// the circuit), and most canonical keys MISS the DB, so caching negative
// results is as valuable as positive ones. On hosts whose RAM is smaller than
// the DB, each avoided lookup is an avoided page fault.
//
// LOOKUP_CACHE_MB caps the approximate value-byte footprint (default 512 MiB;
// 0 disables the cache). When the cap is exceeded the cache is cleared
// wholesale — an epoch reset is cheaper and simpler than LRU bookkeeping and
// the working set refills within seconds.
// ---------------------------------------------------------------------------
// Cache keys are namespaced: byte 0 distinguishes which logical database the
// lookup targeted (shard vs curated); the same 16-byte canonical hash may
// exist in both with different values.
type LookupCacheMap = DashMap<[u8; 17], Option<std::sync::Arc<[u8]>>, rustc_hash::FxBuildHasher>;

pub(crate) const LOOKUP_NS_SHARD: u8 = 0;
pub(crate) const LOOKUP_NS_CURATED: u8 = 1;

static LOOKUP_CACHE_BYTES: AtomicU64 = AtomicU64::new(0);
pub static LOOKUP_CACHE_HITS: AtomicU64 = AtomicU64::new(0);
pub static LOOKUP_CACHE_QUERIES: AtomicU64 = AtomicU64::new(0);

fn lookup_cache_cap_bytes() -> u64 {
    static CAP: OnceLock<u64> = OnceLock::new();
    *CAP.get_or_init(|| {
        std::env::var("LOOKUP_CACHE_MB")
            .ok()
            .and_then(|v| v.parse::<u64>().ok())
            .unwrap_or(512)
            .saturating_mul(1024 * 1024)
    })
}

fn lookup_cache() -> Option<&'static LookupCacheMap> {
    static CACHE: OnceLock<Option<LookupCacheMap>> = OnceLock::new();
    CACHE
        .get_or_init(|| (lookup_cache_cap_bytes() > 0).then(LookupCacheMap::default))
        .as_ref()
}

/// Fetch one namespaced key from the immutable frozen database.
fn raw_db_get(db: &FrozenDb, namespace: u8, key: &[u8; 16]) -> Option<std::sync::Arc<[u8]>> {
    let value = if namespace == LOOKUP_NS_CURATED {
        db.get_curated(key)
    } else {
        db.get_regular(key)
    };
    value.map(std::sync::Arc::from)
}

/// Point lookup with an exact process-wide cache in front. Returns the value
/// bytes (shared, immutable) or None when the key is absent — byte-identical
/// to the uncached lookup on a read-only environment.
pub(crate) fn cached_db_get(
    db: &FrozenDb,
    namespace: u8,
    key: &[u8; 16],
) -> Option<std::sync::Arc<[u8]>> {
    let Some(cache) = lookup_cache() else {
        return raw_db_get(db, namespace, key);
    };
    let mut ns_key = [0u8; 17];
    ns_key[0] = namespace;
    ns_key[1..].copy_from_slice(key);
    LOOKUP_CACHE_QUERIES.fetch_add(1, Ordering::Relaxed);
    if let Some(entry) = cache.get(&ns_key) {
        LOOKUP_CACHE_HITS.fetch_add(1, Ordering::Relaxed);
        return entry.clone();
    }
    let result: Option<std::sync::Arc<[u8]>> = raw_db_get(db, namespace, key);
    let entry_bytes = 17 + 64 + result.as_ref().map_or(0, |v| v.len()) as u64;
    if LOOKUP_CACHE_BYTES.fetch_add(entry_bytes, Ordering::Relaxed) + entry_bytes
        > lookup_cache_cap_bytes()
    {
        cache.clear();
        LOOKUP_CACHE_BYTES.store(entry_bytes, Ordering::Relaxed);
    }
    cache.insert(ns_key, result.clone());
    result
}

fn cached_shard_get(db: &FrozenDb, key: &[u8; 16]) -> Option<std::sync::Arc<[u8]>> {
    cached_db_get(db, LOOKUP_NS_SHARD, key)
}

fn cached_curated_get(db: &FrozenDb, key: &[u8; 16]) -> Option<std::sync::Arc<[u8]>> {
    cached_db_get(db, LOOKUP_NS_CURATED, key)
}

// ---------------------------------------------------------------------------
// Lookup direction strategy.
//
// The shard DBs are keyed by the *minimum* of a circuit's forward and reverse
// canonical polynomial forms (build_from_rocks inserts min(canon_fwd,
// canon_rev) only). Since canonical polys are determined by the function a
// circuit computes, a window's non-min direction key can only exist in the DB
// if it equals the min key. Probing just the min direction is therefore
// exactly equivalent to the legacy forward-then-reverse probe — and it halves
// the number of cold frozen-store probes on misses, which dominate when the DB is
// larger than RAM.
//
// MIN_DIR_LOOKUP=0        -> legacy: forward probe, reverse probe on miss.
// MIN_DIR_LOOKUP=validate -> min-direction probe, but on a miss also probe the
//                            other direction and count/log any hit (which
//                            would disprove the min-key invariant).
// unset / any other value -> min-direction probe only (default).
// ---------------------------------------------------------------------------
#[derive(Clone, Copy, PartialEq, Eq)]
pub(crate) enum MinDirLookup {
    Legacy,
    Min,
    Validate,
}

pub(crate) fn min_dir_lookup_mode() -> MinDirLookup {
    static MODE: OnceLock<MinDirLookup> = OnceLock::new();
    *MODE.get_or_init(|| match std::env::var("MIN_DIR_LOOKUP").as_deref() {
        Ok("0") => MinDirLookup::Legacy,
        Ok("validate") => MinDirLookup::Validate,
        _ => MinDirLookup::Min,
    })
}

pub static MIN_DIR_VALIDATE_PROBES: AtomicU64 = AtomicU64::new(0);
pub static MIN_DIR_VIOLATIONS: AtomicU64 = AtomicU64::new(0);

pub static CANON_TIME: AtomicU64 = AtomicU64::new(0);
pub static CONVEX_FIND_TIME: AtomicU64 = AtomicU64::new(0);
pub static CONVEX_MAX_WIRES_TIME: AtomicU64 = AtomicU64::new(0);
pub static CONVEX_MAX_GATES_TIME: AtomicU64 = AtomicU64::new(0);
pub static CONVEX_SIMPLE_TIME: AtomicU64 = AtomicU64::new(0);
pub static CONTIGUOUS_TIME: AtomicU64 = AtomicU64::new(0);
pub static REWIRE_TIME: AtomicU64 = AtomicU64::new(0);
pub static COMPRESS_TIME: AtomicU64 = AtomicU64::new(0);
pub static REPLACE_TIME: AtomicU64 = AtomicU64::new(0);
pub static DEDUP_TIME: AtomicU64 = AtomicU64::new(0);
pub static CANONICALIZE_TIME: AtomicU64 = AtomicU64::new(0);
pub static CANONICALIZE_TIME_MAX_WIRES: AtomicU64 = AtomicU64::new(0);
pub static CANONICALIZE_TIME_SIMPLE: AtomicU64 = AtomicU64::new(0);
pub static CANONICALIZE_TIME_MAX_GATES: AtomicU64 = AtomicU64::new(0);
pub static FROZEN_LOOKUP_TIME: AtomicU64 = AtomicU64::new(0);
pub static FROM_BLOB_TIME: AtomicU64 = AtomicU64::new(0);
pub static SPLICE_TIME: AtomicU64 = AtomicU64::new(0);
pub static TRIAL_TIME: AtomicU64 = AtomicU64::new(0);

fn compression_trace_enabled() -> bool {
    static ON: OnceLock<bool> = OnceLock::new();
    *ON.get_or_init(|| std::env::var("COMPRESSION_TRACE").is_ok())
}

fn compression_trace_threshold_ms() -> u128 {
    static THRESHOLD: OnceLock<u128> = OnceLock::new();
    *THRESHOLD.get_or_init(|| {
        std::env::var("COMPRESSION_TRACE_MS")
            .ok()
            .and_then(|v| v.parse().ok())
            .unwrap_or(1_000)
    })
}

/// Relative stall rule for `compress_loop` (ported from ssg-gen-mix-clean
/// commit e4d7e33a). When COMPRESS_STALL_FRAC is set, the loop stops once the
/// reduction over the last COMPRESS_STALL_WINDOW sweeps (default 2) falls
/// below `frac * current_size`, floored at the legacy 50 gates. Unset: the
/// legacy `< 50 gates over stable_max sweeps` rule, byte-identical behavior.
fn compress_stall_frac() -> Option<f64> {
    static V: OnceLock<Option<f64>> = OnceLock::new();
    *V.get_or_init(|| {
        std::env::var("COMPRESS_STALL_FRAC")
            .ok()
            .and_then(|v| v.parse().ok())
    })
}

fn compress_stall_window() -> usize {
    static V: OnceLock<usize> = OnceLock::new();
    *V.get_or_init(|| {
        std::env::var("COMPRESS_STALL_WINDOW")
            .ok()
            .and_then(|v| v.parse().ok())
            .unwrap_or(2)
            .max(1)
    })
}

/// Per-chunk wall-clock budget for `compress_big_ancillas` (ported from
/// ssg-gen-mix-clean commit ddc5f584). Past the budget the 100-trial loop
/// stops before the next trial, keeping completed trials; the next sweep
/// re-randomizes chunk boundaries so hard regions are revisited from other
/// angles. Bounds a chunk at budget + one trial; a single in-flight
/// `compress_frozen` call is not interrupted. Unset: unlimited (legacy).
fn compress_chunk_budget_ms() -> Option<u128> {
    static V: OnceLock<Option<u128>> = OnceLock::new();
    *V.get_or_init(|| {
        std::env::var("COMPRESS_CHUNK_BUDGET_MS")
            .ok()
            .and_then(|v| v.parse().ok())
    })
}

pub fn compress_loop(
    circuit: &CircuitSeq,
    n: usize,
    db: &FrozenDb,
    stable_compressions: usize,
    curr_round: usize,
    last_round: usize,
    output_path: &str,
    // Optional Stage-D early stop. Compression still stops earlier if its
    // normal no-progress rule fires at an incompressibility ceiling.
    early_stop_target: Option<usize>,
    tags: &mut Vec<u32>,
) -> CircuitSeq {
    let track = !tags.is_empty();
    let mut acc = circuit.clone();
    let mut acc_tags = tags.clone();
    let mut rng = rand::rng();
    let mut mode = 0usize;
    let stable_compressions = stable_compressions.max(1);
    let stable_max = if last_round > 0 && curr_round == last_round {
        stable_compressions.saturating_mul(2)
    } else {
        stable_compressions
    };
    // Stall rule: with COMPRESS_STALL_FRAC set, watch a COMPRESS_STALL_WINDOW-
    // sweep window against a relative threshold; otherwise the legacy
    // stable_max-sweep window against the absolute 50-gate threshold.
    let stall_frac = compress_stall_frac();
    let stall_window = match stall_frac {
        Some(frac) => {
            println!(
                "[compress] stall rule: stop when < {:.1}% of current size reduced over last {} sweeps",
                frac * 100.0,
                compress_stall_window()
            );
            compress_stall_window()
        }
        None => stable_max,
    };
    if let Some(budget) = compress_chunk_budget_ms() {
        println!("[compress] chunk budget: {} ms per chunk per sweep", budget);
    }
    // Ring buffer of the last stall_window+1 gate counts. Stop when total reduction
    // over the last stall_window iterations is under the threshold.
    let mut recent: std::collections::VecDeque<usize> =
        std::collections::VecDeque::with_capacity(stall_window + 1);
    recent.push_back(acc.gates.len());

    loop {
        let before = acc.gates.len();
        REC_ITER.fetch_add(1, Ordering::Relaxed);

        let max_chunks = 4 * rayon::current_num_threads().max(1);
        let k = if before <= 1500 {
            1
        } else {
            ((before + 1499) / 1500).min(max_chunks)
        };

        let current_mode = [1, 2][mode];
        mode = (mode + 1) % 2;

        let trace = compression_trace_enabled();
        let trace_threshold_ms = compression_trace_threshold_ms();
        let ranges = split_into_random_chunk_ranges(acc.gates.len(), k, &mut rng);
        let acc_tags_ref = &acc_tags;
        let compressed_chunks: Vec<(usize, usize, usize, Vec<[u16; 3]>, Vec<u32>, u128)> = ranges
            .into_par_iter()
            .enumerate()
            .map(|(chunk_idx, (start, end))| {
                let sub = CircuitSeq {
                    gates: acc.gates[start..end].to_vec(),
                };
                let mut chunk_tags = if track {
                    acc_tags_ref[start..end].to_vec()
                } else {
                    Vec::new()
                };
                let chunk_start = Instant::now();
                let gates = compress_big_ancillas(
                    &sub,
                    100,
                    n,
                    db,
                    current_mode,
                    start,
                    &mut chunk_tags,
                )
                .gates;
                let elapsed_ms = chunk_start.elapsed().as_millis();
                if trace && elapsed_ms >= trace_threshold_ms {
                    eprintln!(
                        "[compress-trace] slow chunk mode={} idx={}/{} in_gates={} out_gates={} elapsed_ms={}",
                        current_mode,
                        chunk_idx + 1,
                        k,
                        end - start,
                        gates.len(),
                        elapsed_ms
                    );
                }
                (chunk_idx, end - start, gates.len(), gates, chunk_tags, elapsed_ms)
            })
            .collect();

        let mut compressed_chunks = compressed_chunks;
        compressed_chunks.sort_by_key(|(chunk_idx, _, _, _, _, _)| *chunk_idx);

        if trace {
            let total_chunk_ms: u128 = compressed_chunks
                .iter()
                .map(|(_, _, _, _, _, ms)| *ms)
                .sum();
            if let Some((slow_idx, slow_in, slow_out, _, _, slow_ms)) = compressed_chunks
                .iter()
                .max_by_key(|(_, _, _, _, _, elapsed_ms)| *elapsed_ms)
            {
                eprintln!(
                    "[compress-trace] iteration mode={} chunks={} before={} slowest_idx={} slowest_in={} slowest_out={} slowest_ms={} sum_chunk_ms={}",
                    current_mode,
                    k,
                    before,
                    slow_idx + 1,
                    slow_in,
                    slow_out,
                    slow_ms,
                    total_chunk_ms
                );
            }
        }

        let total_len: usize = compressed_chunks
            .iter()
            .map(|(_, _, out_len, _, _, _)| *out_len)
            .sum();
        let mut new_gates = Vec::with_capacity(total_len);
        let mut new_tags = if track {
            Vec::with_capacity(total_len)
        } else {
            Vec::new()
        };
        for (_, _, _, chunk, chunk_tags, _) in compressed_chunks {
            new_gates.extend(chunk);
            if track {
                new_tags.extend(chunk_tags);
            }
        }

        acc.gates = new_gates;
        if track {
            acc_tags = new_tags;
        }
        let after = acc.gates.len();

        recent.push_back(after);
        if recent.len() > stall_window + 1 {
            recent.pop_front();
        }

        // Stop if the total reduction over the window is under the threshold:
        // relative (frac * current size, floored at 50) when the stall rule is
        // active, the legacy absolute 50 otherwise.
        if recent.len() == stall_window + 1 {
            let window_reduction = recent.front().unwrap().saturating_sub(after);
            match stall_frac {
                Some(frac) => {
                    let threshold = ((frac * after as f64) as usize).max(50);
                    if window_reduction < threshold {
                        println!(
                            "  {}/{}: Early stop — only {} gates reduced over last {} iterations ({} gates, threshold {})",
                            curr_round,
                            last_round,
                            window_reduction,
                            stall_window,
                            after,
                            threshold
                        );
                        break;
                    }
                }
                None => {
                    if window_reduction < 50 {
                        println!(
                            "  {}/{}: Early stop — only {} gates reduced over last {} iterations ({} gates)",
                            curr_round, last_round, window_reduction, stall_window, after
                        );
                        break;
                    }
                }
            }
        }

        if after == before {
            println!("  {}/{}: Stable ({} gates)", curr_round, last_round, after);
        } else {
            println!("  {}/{}: Reduced: {} gates", curr_round, last_round, after);
        }

        if let Some(target) = early_stop_target {
            if after <= target {
                println!(
                    "  {}/{}: Stage-D compression target reached ({} <= {} gates), stopping",
                    curr_round, last_round, after, target
                );
                break;
            }
        }

        // Check if user created write_now
        if std::path::Path::new("write_now").exists() {
            std::fs::remove_file("write_now").ok();
            let mut f = File::create(output_path).expect("create");
            writeln!(f, "{}", acc.repr()).expect("write");
            eprintln!("Wrote {}", output_path);
        }
    }
    if track {
        *tags = acc_tags;
    }
    acc
}

fn merge_expanded_chunks(
    circuit: &CircuitSeq,
    mut chunks: Vec<(usize, usize, Vec<[u16; 3]>)>,
    gate_cap: Option<usize>,
) -> (Vec<[u16; 3]>, usize) {
    chunks.sort_by_key(|(start, _, _)| *start);
    let mut remaining_growth = gate_cap
        .map(|cap| cap.saturating_sub(circuit.gates.len()))
        .unwrap_or(usize::MAX);
    let mut accepted = 0usize;
    let mut new_gates = Vec::with_capacity(
        gate_cap.unwrap_or_else(|| chunks.iter().map(|(_, _, gates)| gates.len()).sum()),
    );

    for (start, end, expanded) in chunks {
        let original = &circuit.gates[start..end];
        if gate_cap.is_none() {
            new_gates.extend(expanded);
            accepted += 1;
            continue;
        }

        // Every chunk replacement is independently equivalent to its original range. Admit only
        // expansions whose positive delta fits the remaining global budget; otherwise retain the
        // original chunk. This makes the size governor functionality-preserving—never truncate an
        // expanded circuit at an arbitrary gate boundary.
        let growth = expanded.len().saturating_sub(original.len());
        if expanded.len() > original.len() && growth <= remaining_growth {
            new_gates.extend(expanded);
            remaining_growth -= growth;
            accepted += 1;
        } else {
            new_gates.extend_from_slice(original);
        }
    }

    (new_gates, accepted)
}

fn expand_once_with_gate_cap(
    circuit: &CircuitSeq,
    n: usize,
    db: &FrozenDb,
    pair_mode: &ExpandPairMode,
    gate_cap: Option<usize>,
) -> CircuitSeq {
    if gate_cap.is_some_and(|cap| circuit.gates.len() >= cap) {
        println!(
            "  Bounded expand skipped: {} gates already at/above cap {}",
            circuit.gates.len(),
            gate_cap.unwrap()
        );
        return circuit.clone();
    }
    let mut rng = rand::rng();
    let before = circuit.gates.len();
    let max_chunks = 4 * rayon::current_num_threads().max(1);
    let k = if before <= 1500 {
        1
    } else {
        ((before + 1499) / 1500).min(max_chunks)
    };
    let ranges = split_into_random_chunk_ranges(before, k, &mut rng);
    let expanded_chunks: Vec<(usize, usize, Vec<[u16; 3]>)> = ranges
        .into_par_iter()
        .map(|(start, end)| {
            let sub = CircuitSeq {
                gates: circuit.gates[start..end].to_vec(),
            };
            (
                start,
                end,
                expand_big_ancillas(&sub, 100, n, db, 0, pair_mode).gates,
            )
        })
        .collect();
    let (new_gates, accepted) = merge_expanded_chunks(circuit, expanded_chunks, gate_cap);
    if let Some(cap) = gate_cap {
        println!(
            "  Bounded expand: {} -> {} gates (cap {}, accepted {} chunks)",
            before,
            new_gates.len(),
            cap,
            accepted
        );
    } else {
        println!("  Expand: {} gates", new_gates.len());
    }
    CircuitSeq { gates: new_gates }
}

/// Single pass of expansion: one round of chunked `expand_big_ancillas` with no loop.
pub fn expand_once(
    circuit: &CircuitSeq,
    n: usize,
    db: &FrozenDb,
    pair_mode: &ExpandPairMode,
) -> CircuitSeq {
    expand_once_with_gate_cap(circuit, n, db, pair_mode, None)
}

/// Functionality-preserving expansion that admits whole equivalent chunks only while their total
/// gate delta fits `gate_cap`. The output is never larger than the cap when the input is not.
pub fn expand_once_bounded(
    circuit: &CircuitSeq,
    n: usize,
    db: &FrozenDb,
    pair_mode: &ExpandPairMode,
    gate_cap: usize,
) -> CircuitSeq {
    expand_once_with_gate_cap(circuit, n, db, pair_mode, Some(gate_cap))
}

pub fn expand_to_gate_factor(
    circuit: &CircuitSeq,
    n: usize,
    db: &FrozenDb,
    pair_mode: &ExpandPairMode,
    factor: usize,
) -> CircuitSeq {
    let (expanded, passes, stalled) = expand_to_gate_factor_once(circuit, n, db, pair_mode, factor);
    print_expand_loop_summary(
        circuit.gates.len(),
        expanded.gates.len(),
        factor,
        passes,
        stalled,
    );
    expanded
}

fn expand_to_gate_factor_once(
    circuit: &CircuitSeq,
    n: usize,
    db: &FrozenDb,
    pair_mode: &ExpandPairMode,
    factor: usize,
) -> (CircuitSeq, usize, bool) {
    let start_len = circuit.gates.len();
    let target_len = start_len.saturating_mul(factor.max(1));
    let mut expanded = circuit.clone();
    let mut passes = 0usize;
    let mut stalled = false;

    while expanded.gates.len() < target_len {
        let before = expanded.gates.len();
        expanded = expand_once(&expanded, n, db, pair_mode);
        passes += 1;
        if expanded.gates.len() <= before {
            stalled = true;
            break;
        }
    }

    (expanded, passes, stalled)
}

fn print_expand_loop_summary(
    start_len: usize,
    expanded_len: usize,
    factor: usize,
    passes: usize,
    stalled: bool,
) {
    let target_len = start_len.saturating_mul(factor.max(1));
    if stalled {
        println!(
            "  Expand loop stalled: {} -> {} gates (target {}, passes {})",
            start_len, expanded_len, target_len, passes
        );
    }
    println!(
        "  Expand loop: {} -> {} gates (target {}, passes {})",
        start_len, expanded_len, target_len, passes
    );
}

// Expand with ancilla wires or gates
/// Selects which method to use when a 2-gate subcircuit is sampled in the expand functions.
/// Larger subcircuits always use the curated frozen-store expansion path.
pub enum ExpandPairMode {
    /// Use the curated frozen store to find a longer equivalent pair.
    Curated,
    /// Skip the specialized pair helper and use the generic curated expansion
    /// lookup for 2-gate subcircuits too. Database routing remains curated.
    GenericCurated,
}

pub fn expand_frozen(
    c: &CircuitSeq,
    trials: usize,
    n: usize,
    db: &FrozenDb,
    pair_mode: &ExpandPairMode,
) -> CircuitSeq {
    use crate::circuit::circuit::polys_repr_blob;
    use xxhash_rust::xxh3::xxh3_128;

    assert!(
        db.has_curated(),
        "frozen expansion requires FROZEN_CURATED_DIR"
    );

    let mut expanded = c.clone();

    if expanded.gates.is_empty() {
        return CircuitSeq { gates: Vec::new() };
    }

    let mut rng = rand::rng();

    for _ in 0..trials {
        let t_trial = Instant::now();

        let (sub, start, end) = random_subcircuit_max(&expanded, 5);

        // Require at least 2 gates; 1-gate subcircuits cannot be expanded meaningfully.
        if sub.gates.len() < 2 {
            continue;
        }

        // --- 2-gate path: use the curated pair lookup when requested ---
        if sub.gates.len() == 2 {
            match pair_mode {
                ExpandPairMode::Curated => {
                    use crate::replace::pairs::expand_curated_frozen;
                    if let Some(repl) = expand_curated_frozen(&sub.gates, n, db) {
                        if repl.len() > 2 {
                            expanded.gates.splice(start..end, repl);
                        }
                    }
                    TRIAL_TIME.fetch_add(t_trial.elapsed().as_nanos() as u64, Ordering::Relaxed);
                    continue;
                }
                ExpandPairMode::GenericCurated => { /* fall through to generic curated lookup */ }
            }
        }

        let t_canon = Instant::now();
        let (fwd_polys, fwd_order, used) = sub.canonicalize_polys_single(false);
        CANONICALIZE_TIME.fetch_add(t_canon.elapsed().as_nanos() as u64, Ordering::Relaxed);

        if fwd_polys.is_empty() {
            continue;
        }

        // Curated entries are keyed by the forward canonical form. Expansion
        // must never consult the regular compression store.
        let fwd_key = xxh3_128(&polys_repr_blob(&fwd_polys)).to_le_bytes();
        let t_lookup = Instant::now();
        let lookup_result = cached_curated_get(db, &fwd_key);
        FROZEN_LOOKUP_TIME.fetch_add(t_lookup.elapsed().as_nanos() as u64, Ordering::Relaxed);
        let value = match lookup_result {
            Some(value) => value,
            None => continue,
        };
        let final_order = fwd_order;
        let is_reversed = false;

        let t_blob = Instant::now();
        let mut candidates: Vec<CircuitSeq> = Vec::new();
        let mut pos = 0;
        while pos < value.len() {
            if pos + 1 > value.len() {
                break;
            }
            let len = value[pos] as usize;
            pos += 1;
            if pos + len > value.len() {
                break;
            }
            let candidate = CircuitSeq::from_blob(&value[pos..pos + len]);
            pos += len;
            if candidate.gates.len() > sub.gates.len() {
                candidates.push(candidate);
            }
        }
        FROM_BLOB_TIME.fetch_add(t_blob.elapsed().as_nanos() as u64, Ordering::Relaxed);

        if candidates.is_empty() {
            continue;
        }

        let max_gates = candidates
            .iter()
            .map(|candidate| candidate.gates.len())
            .max()
            .unwrap();
        let best: Vec<CircuitSeq> = candidates
            .into_iter()
            .filter(|candidate| candidate.gates.len() == max_gates)
            .collect();
        let t_rewire = Instant::now();
        let mut mapped: Vec<CircuitSeq> = best
            .into_iter()
            .map(|candidate| {
                candidate_to_circuit_space(candidate, is_reversed, &final_order, &used, n, &mut rng)
            })
            .collect();
        REWIRE_TIME.fetch_add(t_rewire.elapsed().as_nanos() as u64, Ordering::Relaxed);

        // Rank in circuit wire space against the real surrounding gates, so leeway/fanout
        // features are measured on the wires the replacement will actually sit between.
        let idx = if incoming_rank_mode() == IncomingRankMode::Fanout {
            fanout_pick_index(&mapped, &expanded.gates[..start], &expanded.gates[end..])
        } else {
            rng.random_range(0..mapped.len())
        };
        let repl = mapped.swap_remove(idx);
        note_wire_use(&repl.gates);

        let t_splice = Instant::now();
        expanded.gates.splice(start..end, repl.gates);
        SPLICE_TIME.fetch_add(t_splice.elapsed().as_nanos() as u64, Ordering::Relaxed);

        TRIAL_TIME.fetch_add(t_trial.elapsed().as_nanos() as u64, Ordering::Relaxed);
    }

    expanded
}

pub fn compress_frozen(
    c: &CircuitSeq,
    trials: usize,
    n: usize,
    db: &FrozenDb,
    mode: usize,
    base_offset: usize,
    tags: &mut Vec<u32>,
) -> CircuitSeq {
    use crate::circuit::circuit::polys_repr_blob;
    use xxhash_rust::xxh3::xxh3_128;

    let track = !tags.is_empty();
    let mut compressed = c.clone();

    let mut j = 0;
    while j < compressed.gates.len().saturating_sub(1) {
        if compressed.gates[j] == compressed.gates[j + 1] {
            compressed.gates.drain(j..=j + 1);
            if track {
                tags.drain(j..=j + 1);
            }
            j = j.saturating_sub(2);
        } else {
            j += 1;
        }
    }

    if compressed.gates.is_empty() {
        return CircuitSeq { gates: Vec::new() };
    }

    let mut rng = rand::rng();

    for _ in 0..trials {
        let t_trial = Instant::now();

        let (sub, start, end) = random_subcircuit(&compressed);

        if sub.gates.is_empty() {
            continue;
        }

        let t_canon = Instant::now();
        let (fwd_polys, fwd_order, used) = sub.canonicalize_polys_single(false);
        let canon_elapsed = t_canon.elapsed().as_nanos() as u64;
        CANONICALIZE_TIME.fetch_add(canon_elapsed, Ordering::Relaxed);
        match mode {
            0 => CANONICALIZE_TIME_MAX_WIRES.fetch_add(canon_elapsed, Ordering::Relaxed),
            2 => CANONICALIZE_TIME_MAX_GATES.fetch_add(canon_elapsed, Ordering::Relaxed),
            _ => CANONICALIZE_TIME_SIMPLE.fetch_add(canon_elapsed, Ordering::Relaxed),
        };
        if compression_trace_enabled()
            && (canon_elapsed as u128 / 1_000_000) >= compression_trace_threshold_ms()
        {
            eprintln!(
                "[compress-trace] slow inner-canon mode={} direction=forward parent_gates={} inner_start={} inner_end={} inner_gates={} inner_wires={} elapsed_ms={}",
                mode,
                compressed.gates.len(),
                start,
                end,
                sub.gates.len(),
                sub.used_wires().len(),
                canon_elapsed as u128 / 1_000_000
            );
        }

        if fwd_polys.is_empty() {
            continue;
        }

        if attempt_stats_enabled() {
            *ATTEMPT_HISTOGRAM
                .entry((sub.gates.len() as u8, used.len() as u8))
                .or_insert(0) += 1;
        }

        let lookup_mode = min_dir_lookup_mode();

        // Reverse-direction canonicalization, shared by the legacy fallback
        // path and the min-direction path; keeps the existing per-mode timers.
        let canonicalize_rev = |sub: &CircuitSeq,
                                compressed_len: usize|
         -> (Vec<crate::circuit::circuit::Polynomial>, Permutation) {
            let t_canon2 = Instant::now();
            let (rev_polys, rev_order, _) = sub.canonicalize_polys_single(true);
            let canon2_elapsed = t_canon2.elapsed().as_nanos() as u64;
            CANONICALIZE_TIME.fetch_add(canon2_elapsed, Ordering::Relaxed);
            match mode {
                0 => CANONICALIZE_TIME_MAX_WIRES.fetch_add(canon2_elapsed, Ordering::Relaxed),
                2 => CANONICALIZE_TIME_MAX_GATES.fetch_add(canon2_elapsed, Ordering::Relaxed),
                _ => CANONICALIZE_TIME_SIMPLE.fetch_add(canon2_elapsed, Ordering::Relaxed),
            };
            if compression_trace_enabled()
                && (canon2_elapsed as u128 / 1_000_000) >= compression_trace_threshold_ms()
            {
                eprintln!(
                    "[compress-trace] slow inner-canon mode={} direction=reverse parent_gates={} inner_start={} inner_end={} inner_gates={} inner_wires={} elapsed_ms={}",
                    mode,
                    compressed_len,
                    start,
                    end,
                    sub.gates.len(),
                    sub.used_wires().len(),
                    canon2_elapsed as u128 / 1_000_000
                );
            }
            (rev_polys, rev_order)
        };

        let (value, final_order, is_reversed) = if lookup_mode == MinDirLookup::Legacy {
            let fwd_key = xxh3_128(&polys_repr_blob(&fwd_polys)).to_le_bytes();
            let t_lookup = Instant::now();
            let fwd_result = cached_shard_get(db, &fwd_key);
            FROZEN_LOOKUP_TIME.fetch_add(t_lookup.elapsed().as_nanos() as u64, Ordering::Relaxed);

            if let Some(v) = fwd_result {
                (v, fwd_order, false)
            } else {
                let (rev_polys, rev_order) = canonicalize_rev(&sub, compressed.gates.len());

                if rev_polys.is_empty() {
                    continue;
                }

                let rev_key = xxh3_128(&polys_repr_blob(&rev_polys)).to_le_bytes();
                let t_lookup2 = Instant::now();
                let rev_result = cached_shard_get(db, &rev_key);
                FROZEN_LOOKUP_TIME
                    .fetch_add(t_lookup2.elapsed().as_nanos() as u64, Ordering::Relaxed);

                match rev_result {
                    Some(v) => (v, rev_order, true),
                    None => continue,
                }
            }
        } else {
            // Min-direction probe (see MinDirLookup above): probe only the
            // direction whose canonical form a min-keyed DB can contain.
            let (rev_polys, rev_order) = canonicalize_rev(&sub, compressed.gates.len());

            if rev_polys.is_empty() {
                continue;
            }

            let rev_is_min = rev_polys < fwd_polys;
            let (min_polys, min_order, min_reversed, alt_polys, alt_order, alt_reversed) =
                if rev_is_min {
                    (rev_polys, rev_order, true, fwd_polys, fwd_order, false)
                } else {
                    (fwd_polys, fwd_order, false, rev_polys, rev_order, true)
                };

            let min_key = xxh3_128(&polys_repr_blob(&min_polys)).to_le_bytes();
            let t_lookup = Instant::now();
            let min_result = cached_shard_get(db, &min_key);
            FROZEN_LOOKUP_TIME.fetch_add(t_lookup.elapsed().as_nanos() as u64, Ordering::Relaxed);

            if let Some(v) = min_result {
                (v, min_order, min_reversed)
            } else if lookup_mode == MinDirLookup::Validate {
                MIN_DIR_VALIDATE_PROBES.fetch_add(1, Ordering::Relaxed);
                let alt_key = xxh3_128(&polys_repr_blob(&alt_polys)).to_le_bytes();
                let t_lookup2 = Instant::now();
                let alt_result = cached_shard_get(db, &alt_key);
                FROZEN_LOOKUP_TIME
                    .fetch_add(t_lookup2.elapsed().as_nanos() as u64, Ordering::Relaxed);
                match alt_result {
                    Some(v) => {
                        MIN_DIR_VIOLATIONS.fetch_add(1, Ordering::Relaxed);
                        eprintln!(
                            "[min-dir-violation] compress: non-min canonical key present while min key absent (gates={})",
                            sub.gates.len()
                        );
                        (v, alt_order, alt_reversed)
                    }
                    None => continue,
                }
            } else {
                continue;
            }
        };

        let t_blob = Instant::now();
        let mut candidates: Vec<CircuitSeq> = Vec::new();
        let mut pos = 0;
        while pos < value.len() {
            if pos + 1 > value.len() {
                break;
            }
            let len = value[pos] as usize;
            pos += 1;
            if pos + len > value.len() {
                break;
            }
            let candidate = CircuitSeq::from_blob(&value[pos..pos + len]);
            pos += len;
            if candidate.gates.len() < sub.gates.len() {
                candidates.push(candidate);
            }
        }
        FROM_BLOB_TIME.fetch_add(t_blob.elapsed().as_nanos() as u64, Ordering::Relaxed);

        if attempt_stats_enabled() {
            *ATTEMPT_HIT_HISTOGRAM
                .entry((sub.gates.len() as u8, used.len() as u8))
                .or_insert(0) += 1;
            if !candidates.is_empty() {
                *ATTEMPT_SHORTER_HISTOGRAM
                    .entry((sub.gates.len() as u8, used.len() as u8))
                    .or_insert(0) += 1;
            }
        }

        if candidates.is_empty() {
            continue;
        }

        let min_gates = candidates.iter().map(|c| c.gates.len()).min().unwrap();
        let incoming_mode = incoming_rank_mode();
        let best: Vec<CircuitSeq> = candidates
            .into_iter()
            .filter(|candidate| candidate.gates.len() == min_gates)
            .collect();
        let t_rewire = Instant::now();
        let mut mapped: Vec<CircuitSeq> = best
            .into_iter()
            .map(|candidate| {
                candidate_to_circuit_space(candidate, is_reversed, &final_order, &used, n, &mut rng)
            })
            .collect();
        REWIRE_TIME.fetch_add(t_rewire.elapsed().as_nanos() as u64, Ordering::Relaxed);

        // Rank in circuit wire space: previously canonical-space candidate labels were
        // compared against circuit-space context, making cross-boundary features garbage.
        let pick = if incoming_mode == IncomingRankMode::Fanout {
            fanout_pick_index(
                &mapped,
                &compressed.gates[..start],
                &compressed.gates[end..],
            )
        } else {
            rng.random_range(0..mapped.len())
        };
        let repl = mapped.swap_remove(pick);
        note_wire_use(&repl.gates);
        *COMPRESSION_HISTOGRAM
            .entry((sub.gates.len() as u8, repl.gates.len() as u8))
            .or_insert(0) += 1;

        if record_enabled() {
            let ws: Vec<u16> = repl.gates.iter().flatten().copied().collect();
            record_replacement(
                "compress",
                REC_ITER.load(Ordering::Relaxed),
                base_offset + start,
                base_offset + start + repl.gates.len(),
                end - start,
                &ws,
            );
        }

        let t_splice = Instant::now();
        let repl_len = repl.gates.len();
        if repl_len == end - start {
            compressed.gates[start..end].copy_from_slice(&repl.gates);
        } else {
            compressed.gates.splice(start..end, repl.gates);
        }
        if track {
            let nt = new_gate_tag(&tags[start..end]);
            tags.splice(start..end, std::iter::repeat(nt).take(repl_len));
        }
        SPLICE_TIME.fetch_add(t_splice.elapsed().as_nanos() as u64, Ordering::Relaxed);

        TRIAL_TIME.fetch_add(t_trial.elapsed().as_nanos() as u64, Ordering::Relaxed);
    }

    let mut j = 0;
    while j < compressed.gates.len().saturating_sub(1) {
        if compressed.gates[j] == compressed.gates[j + 1] {
            compressed.gates.drain(j..=j + 1);
            if track {
                tags.drain(j..=j + 1);
            }
            j = j.saturating_sub(2);
        } else {
            j += 1;
        }
    }

    compressed
}

pub fn compress_big_ancillas(
    c: &CircuitSeq,
    trials: usize,
    num_wires: usize,
    db: &FrozenDb,
    mode: usize,
    base_offset: usize,
    tags: &mut Vec<u32>,
) -> CircuitSeq {
    let track = !tags.is_empty();
    let mut circuit = c.clone();
    let mut rng = rand::rng();

    let mut j = 0;
    while j < circuit.gates.len().saturating_sub(1) {
        if circuit.gates[j] == circuit.gates[j + 1] {
            circuit.gates.drain(j..=j + 1);
            if track {
                tags.drain(j..=j + 1);
            }
            j = j.saturating_sub(2);
        } else {
            j += 1;
        }
    }

    // Wall-clock budget per chunk (COMPRESS_CHUNK_BUDGET_MS): checked between
    // trials, so a chunk is bounded at budget + one trial. Completed trials are
    // kept and the trailing dedup below still runs; the next sweep re-randomizes
    // chunk boundaries so budget-hit regions get revisited from other angles.
    let chunk_budget = compress_chunk_budget_ms();
    let chunk_clock = Instant::now();

    for _ in 0..trials {
        if let Some(budget_ms) = chunk_budget {
            if chunk_clock.elapsed().as_millis() >= budget_ms {
                break;
            }
        }
        let t0 = Instant::now();
        let (mut subcircuit_gates, _) = match mode {
            0 => find_convex_subcircuit_max_wires(30, num_wires, &circuit, &mut rng),
            2 => find_convex_subcircuit_max_gates(21, num_wires, &circuit, &mut rng),
            _ => simple_find_convex_subcircuit(num_wires, &circuit, &mut rng),
        };
        let elapsed = t0.elapsed().as_nanos() as u64;
        CONVEX_FIND_TIME.fetch_add(elapsed, Ordering::Relaxed);
        match mode {
            0 => CONVEX_MAX_WIRES_TIME.fetch_add(elapsed, Ordering::Relaxed),
            2 => CONVEX_MAX_GATES_TIME.fetch_add(elapsed, Ordering::Relaxed),
            _ => CONVEX_SIMPLE_TIME.fetch_add(elapsed, Ordering::Relaxed),
        };

        if subcircuit_gates.is_empty() {
            continue;
        }

        let gates: Vec<[u16; 3]> = subcircuit_gates.iter().map(|&g| circuit.gates[g]).collect();
        subcircuit_gates.sort();

        let t1 = Instant::now();
        let cc = contiguous_convex(&mut circuit, &mut subcircuit_gates, num_wires, tags);
        CONTIGUOUS_TIME.fetch_add(t1.elapsed().as_nanos() as u64, Ordering::Relaxed);
        let (start, end) = match cc {
            Some(se) => se,
            None => continue,
        };

        let mut subcircuit = CircuitSeq { gates };

        let mut used_wires = subcircuit.used_wires();
        let mut used_wire_mask = vec![false; num_wires];
        for &wire in &used_wires {
            if let Some(slot) = used_wire_mask.get_mut(wire as usize) {
                *slot = true;
            }
        }
        let n_wires = used_wires.len();
        let max = num_wires;
        let new_wires = rng.random_range(n_wires..=max);
        if new_wires > n_wires {
            let mut count = n_wires;
            while count < new_wires {
                let random = rng.random_range(0..num_wires);
                if used_wire_mask[random] {
                    continue;
                }
                used_wire_mask[random] = true;
                used_wires.push(random as u16);
                count += 1;
            }
        }

        let sub_num_wires = used_wires.len();

        let t4 = Instant::now();
        let sub_gates = subcircuit.gates.len();
        let mut block_tags = if track {
            tags[start..=end].to_vec()
        } else {
            Vec::new()
        };
        let subcircuit_temp = compress_frozen(
            &subcircuit,
            10,
            sub_num_wires,
            db,
            mode,
            base_offset + start,
            &mut block_tags,
        );
        let compress_elapsed = t4.elapsed();
        COMPRESS_TIME.fetch_add(compress_elapsed.as_nanos() as u64, Ordering::Relaxed);
        if compression_trace_enabled()
            && compress_elapsed.as_millis() >= compression_trace_threshold_ms()
        {
            eprintln!(
                "[compress-trace] slow compress_frozen mode={} outer_gates={} outer_wires={} outer_span={} out_gates={} elapsed_ms={}",
                mode,
                sub_gates,
                sub_num_wires,
                end - start + 1,
                subcircuit_temp.gates.len(),
                compress_elapsed.as_millis()
            );
        }

        subcircuit = subcircuit_temp;

        let t6 = Instant::now();
        let repl_len = subcircuit.gates.len();
        let old_len = end - start + 1;

        if repl_len == old_len {
            for i in 0..repl_len {
                circuit.gates[start + i] = subcircuit.gates[i];
            }
            if track {
                tags[start..start + repl_len].copy_from_slice(&block_tags);
            }
        } else if repl_len < old_len {
            for i in 0..repl_len {
                circuit.gates[start + i] = subcircuit.gates[i];
            }
            for i in (end + 1)..circuit.gates.len() {
                circuit.gates[i - (old_len - repl_len)] = circuit.gates[i];
            }
            circuit
                .gates
                .truncate(circuit.gates.len() - (old_len - repl_len));
            if track {
                for i in 0..repl_len {
                    tags[start + i] = block_tags[i];
                }
                for i in (end + 1)..tags.len() {
                    tags[i - (old_len - repl_len)] = tags[i];
                }
                tags.truncate(tags.len() - (old_len - repl_len));
            }
        } else {
            panic!("Replacement grew, which is not allowed");
        }
        REPLACE_TIME.fetch_add(t6.elapsed().as_nanos() as u64, Ordering::Relaxed);
    }

    let t7 = Instant::now();
    let mut j = 0;
    while j < circuit.gates.len().saturating_sub(1) {
        if circuit.gates[j] == circuit.gates[j + 1] {
            circuit.gates.drain(j..=j + 1);
            if track {
                tags.drain(j..=j + 1);
            }
            j = j.saturating_sub(2);
        } else {
            j += 1;
        }
    }
    DEDUP_TIME.fetch_add(t7.elapsed().as_nanos() as u64, Ordering::Relaxed);

    circuit
}

pub fn expand_big_ancillas(
    c: &CircuitSeq,
    trials: usize,
    num_wires: usize,
    db: &FrozenDb,
    mode: usize,
    pair_mode: &ExpandPairMode,
) -> CircuitSeq {
    let mut circuit = c.clone();
    let mut rng = rand::rng();

    for _ in 0..trials {
        let t0 = Instant::now();
        let (mut subcircuit_gates, _) = match mode {
            0 => find_convex_subcircuit_max_wires(30, num_wires, &circuit, &mut rng),
            2 => find_convex_subcircuit_max_gates(21, num_wires, &circuit, &mut rng),
            _ => simple_find_convex_subcircuit(num_wires, &circuit, &mut rng),
        };
        let elapsed = t0.elapsed().as_nanos() as u64;
        CONVEX_FIND_TIME.fetch_add(elapsed, Ordering::Relaxed);
        match mode {
            0 => CONVEX_MAX_WIRES_TIME.fetch_add(elapsed, Ordering::Relaxed),
            2 => CONVEX_MAX_GATES_TIME.fetch_add(elapsed, Ordering::Relaxed),
            _ => CONVEX_SIMPLE_TIME.fetch_add(elapsed, Ordering::Relaxed),
        };

        if subcircuit_gates.is_empty() {
            continue;
        }

        let gates: Vec<[u16; 3]> = subcircuit_gates.iter().map(|&g| circuit.gates[g]).collect();

        if gates.len() < 2 {
            continue;
        }

        subcircuit_gates.sort();

        let t1 = Instant::now();
        let cc = contiguous_convex(
            &mut circuit,
            &mut subcircuit_gates,
            num_wires,
            &mut Vec::new(),
        );
        CONTIGUOUS_TIME.fetch_add(t1.elapsed().as_nanos() as u64, Ordering::Relaxed);
        let (start, end) = match cc {
            Some(se) => se,
            None => continue,
        };

        let subcircuit = CircuitSeq { gates };

        // --- 2-gate path: use pair functions directly on circuit wire values ---
        if subcircuit.gates.len() == 2 {
            let t6 = Instant::now();
            let repl_opt: Option<Vec<[u16; 3]>> = match pair_mode {
                ExpandPairMode::Curated => {
                    use crate::replace::pairs::expand_curated_frozen;
                    expand_curated_frozen(
                        &[circuit.gates[start], circuit.gates[end]],
                        num_wires,
                        db,
                    )
                }
                ExpandPairMode::GenericCurated => None, // generic curated lookup below
            };
            if let Some(repl) = repl_opt {
                if repl.len() > 2 {
                    circuit.gates.splice(start..=end, repl);
                }
                REPLACE_TIME.fetch_add(t6.elapsed().as_nanos() as u64, Ordering::Relaxed);
                continue;
            }
            // Regular mode falls through to expand_frozen.
            REPLACE_TIME.fetch_add(t6.elapsed().as_nanos() as u64, Ordering::Relaxed);
        }

        // --- 3-5 gate path (and 2-gate DB/miss path): use expand_frozen. Keep pair_mode so
        // any 2-gate subproblem sampled inside expand_frozen can still use curated pairs.
        // Pass num_wires (full circuit wire count) so extra wires are assigned correctly.
        let t4 = Instant::now();
        let subcircuit_temp = expand_frozen(&subcircuit, 10, num_wires, db, pair_mode);
        COMPRESS_TIME.fetch_add(t4.elapsed().as_nanos() as u64, Ordering::Relaxed);

        let t6 = Instant::now();
        let repl_len = subcircuit_temp.gates.len();
        let old_len = end - start + 1;

        if repl_len > old_len {
            circuit.gates.splice(start..=end, subcircuit_temp.gates);
        }
        REPLACE_TIME.fetch_add(t6.elapsed().as_nanos() as u64, Ordering::Relaxed);
    }

    circuit
}

// For timing and benchmarking purposes
pub fn print_compress_timers() {
    use crate::circuit::circuit::{
        CANON4_RULE_L_BRANCHES, CANON4_RULE_L_CALLS, CANON4_RULE_L_TIME,
    };
    use crate::replace::transpositions::{SAMF_COMPRESSIONS_FAILED, SAMF_COMPRESSIONS_MADE};

    let canon = CANON_TIME.load(Ordering::Relaxed);
    let rewire = REWIRE_TIME.load(Ordering::Relaxed);
    let convex_find = CONVEX_FIND_TIME.load(Ordering::Relaxed);
    let convex_max_wires = CONVEX_MAX_WIRES_TIME.load(Ordering::Relaxed);
    let convex_max_gates = CONVEX_MAX_GATES_TIME.load(Ordering::Relaxed);
    let convex_simple = CONVEX_SIMPLE_TIME.load(Ordering::Relaxed);
    let contiguous = CONTIGUOUS_TIME.load(Ordering::Relaxed);
    let replace = REPLACE_TIME.load(Ordering::Relaxed);
    let dedup = DEDUP_TIME.load(Ordering::Relaxed);
    let canonicalize = CANONICALIZE_TIME.load(Ordering::Relaxed);
    let canon_max_wires = CANONICALIZE_TIME_MAX_WIRES.load(Ordering::Relaxed);
    let canon_simple = CANONICALIZE_TIME_SIMPLE.load(Ordering::Relaxed);
    let canon_max_gates = CANONICALIZE_TIME_MAX_GATES.load(Ordering::Relaxed);
    let frozen_lookup = FROZEN_LOOKUP_TIME.load(Ordering::Relaxed);
    let from_blob = FROM_BLOB_TIME.load(Ordering::Relaxed);
    let trial = TRIAL_TIME.load(Ordering::Relaxed);
    let rule_l_time = CANON4_RULE_L_TIME.load(Ordering::Relaxed);
    let rule_l_calls = CANON4_RULE_L_CALLS.load(Ordering::Relaxed);
    let rule_l_branches = CANON4_RULE_L_BRANCHES.load(Ordering::Relaxed);

    let samf_made = SAMF_COMPRESSIONS_MADE.load(Ordering::Relaxed);
    let samf_failed = SAMF_COMPRESSIONS_FAILED.load(Ordering::Relaxed);

    let ns = 60_000_000_000.0f64;
    // 15 minutes by default; COMPRESS_TIMERS_MIN_MINUTES overrides so short
    // benchmark runs can print the full breakdown (e.g. =0 prints everything).
    let threshold_minutes = std::env::var("COMPRESS_TIMERS_MIN_MINUTES")
        .ok()
        .and_then(|v| v.parse::<f64>().ok())
        .unwrap_or(15.0);
    let threshold_ns = threshold_minutes * ns;

    println!("--- Compression Timing Totals (minutes) ---");
    if canon as f64 >= threshold_ns {
        println!("Canonicalization time: {:.2} min", canon as f64 / ns);
    }
    if rewire as f64 >= threshold_ns {
        println!("Rewire subcircuit time: {:.2} min", rewire as f64 / ns);
    }
    if convex_find as f64 >= threshold_ns {
        println!(
            "Convex subcircuit find time: {:.2} min",
            convex_find as f64 / ns
        );
        if convex_max_wires as f64 >= threshold_ns {
            println!("  max_wires: {:.2} min", convex_max_wires as f64 / ns);
        }
        if convex_max_gates as f64 >= threshold_ns {
            println!("  max_gates: {:.2} min", convex_max_gates as f64 / ns);
        }
        if convex_simple as f64 >= threshold_ns {
            println!("  simple:    {:.2} min", convex_simple as f64 / ns);
        }
    }
    if contiguous as f64 >= threshold_ns {
        println!(
            "Contiguous convex subcircuit time: {:.2} min",
            contiguous as f64 / ns
        );
    }
    if replace as f64 >= threshold_ns {
        println!("Replacement time: {:.2} min", replace as f64 / ns);
    }
    if dedup as f64 >= threshold_ns {
        println!("Deduplication time: {:.2} min", dedup as f64 / ns);
    }
    if canonicalize as f64 >= threshold_ns {
        println!(
            "Subcircuit canonicalize time: {:.2} min",
            canonicalize as f64 / ns
        );
        if canon_max_wires as f64 >= threshold_ns {
            println!("  max_wires: {:.2} min", canon_max_wires as f64 / ns);
        }
        if canon_max_gates as f64 >= threshold_ns {
            println!("  max_gates: {:.2} min", canon_max_gates as f64 / ns);
        }
        if canon_simple as f64 >= threshold_ns {
            println!("  simple:    {:.2} min", canon_simple as f64 / ns);
        }
    }
    if frozen_lookup as f64 >= threshold_ns {
        println!("Frozen lookup time: {:.2} min", frozen_lookup as f64 / ns);
    }
    let cache_queries = LOOKUP_CACHE_QUERIES.load(Ordering::Relaxed);
    if cache_queries > 0 {
        let cache_hits = LOOKUP_CACHE_HITS.load(Ordering::Relaxed);
        println!(
            "Lookup cache: {} queries, {} hits ({:.1}%)",
            cache_queries,
            cache_hits,
            100.0 * cache_hits as f64 / cache_queries as f64
        );
    }
    let canon_queries = crate::circuit::circuit::CANON_CACHE_QUERIES.load(Ordering::Relaxed);
    if canon_queries > 0 {
        let canon_hits = crate::circuit::circuit::CANON_CACHE_HITS.load(Ordering::Relaxed);
        println!(
            "Canon cache: {} queries, {} hits ({:.1}%)",
            canon_queries,
            canon_hits,
            100.0 * canon_hits as f64 / canon_queries as f64
        );
    }
    let min_dir_probes = MIN_DIR_VALIDATE_PROBES.load(Ordering::Relaxed);
    if min_dir_probes > 0 {
        println!(
            "Min-dir validation: {} alt probes, {} violations",
            min_dir_probes,
            MIN_DIR_VIOLATIONS.load(Ordering::Relaxed)
        );
    }
    if from_blob as f64 >= threshold_ns {
        println!(
            "CircuitSeq from_blob time: {:.2} min",
            from_blob as f64 / ns
        );
    }
    if trial as f64 >= threshold_ns {
        println!("Trial loop time: {:.2} min", trial as f64 / ns);
    }
    if rule_l_time as f64 >= threshold_ns || std::env::var("COMPRESSION_TRACE").is_ok() {
        let seconds = rule_l_time as f64 / 1e9;
        let avg_ms = if rule_l_calls == 0 {
            0.0
        } else {
            rule_l_time as f64 / 1e6 / rule_l_calls as f64
        };
        let avg_branches = if rule_l_calls == 0 {
            0.0
        } else {
            rule_l_branches as f64 / rule_l_calls as f64
        };
        println!(
            "Rule L time: {:.2} s  calls: {}  avg_ms: {:.2}  avg_branches: {:.2}",
            seconds, rule_l_calls, avg_ms, avg_branches
        );
    }
    println!(
        "SAMF compressions made: {}  failed: {}",
        samf_made, samf_failed
    );

    print_attempt_stats();

    if std::env::var("BENCH_CANON").is_ok() {
        use crate::circuit::circuit::{CANON_BENCH_CALLS, CANON4_CORE_TIME, POLYCANON_CORE_TIME};

        let c4 = CANON4_CORE_TIME.load(Ordering::Relaxed) as f64;
        let pc = POLYCANON_CORE_TIME.load(Ordering::Relaxed) as f64;
        let calls = CANON_BENCH_CALLS.load(Ordering::Relaxed).max(1) as f64;
        println!("--- Canonicalization benchmark (BENCH_CANON) ---");
        println!("matched calls:    {}", calls as u64);
        println!(
            "canon4 total:     {:.3} s   ({:.1} us/call)",
            c4 / 1e9,
            c4 / 1e3 / calls
        );
        println!(
            "polycanon total:  {:.3} s   ({:.1} us/call)",
            pc / 1e9,
            pc / 1e3 / calls
        );
        println!("ratio poly/canon4:{:.2}x", pc / c4.max(1.0));
    }
}

#[cfg(test)]
mod tests {
    use super::{
        cand_features, candidate_to_circuit_space, merge_expanded_chunks, order_coldest_first,
    };
    use crate::circuit::circuit::{CircuitSeq, Permutation};

    #[test]
    fn coldest_first_orders_by_load_with_stable_tiebreak() {
        // Pre-shuffled input; loads: wire 4 hot, wires 1/9 cold, wires 2/7 middling.
        let mut wires = vec![4u16, 9, 2, 1, 7];
        let loads = |w: u16| match w {
            4 => 100u32,
            2 | 7 => 5,
            _ => 0,
        };
        order_coldest_first(&mut wires, loads);

        assert_eq!(
            wires,
            vec![9, 1, 2, 7, 4],
            "cold wires first, hot last, equal loads keep prior (shuffled) order"
        );
    }

    fn candidate() -> CircuitSeq {
        CircuitSeq {
            gates: vec![[0, 1, 2], [2, 0, 1], [1, 2, 0]],
        }
    }

    #[test]
    fn bounded_expansion_admits_only_whole_equivalent_chunks() {
        let circuit = CircuitSeq {
            gates: vec![[0, 1, 2], [1, 2, 3], [2, 3, 4], [3, 4, 5]],
        };
        let large = vec![[9, 9, 9]; 5]; // +3 gates; does not fit the two-gate budget.
        let small = vec![[8, 8, 8]; 3]; // +1 gate; does fit.

        let (merged, accepted) = merge_expanded_chunks(
            &circuit,
            vec![(0, 2, large), (2, 4, small.clone())],
            Some(6),
        );

        assert_eq!(accepted, 1);
        assert_eq!(&merged[..2], &circuit.gates[..2]);
        assert_eq!(&merged[2..], small.as_slice());
        assert_eq!(merged.len(), 5);
    }

    #[test]
    fn bounded_expansion_accepts_exact_cap() {
        let circuit = CircuitSeq {
            gates: vec![[0, 1, 2], [1, 2, 3]],
        };
        let exact = vec![[7, 7, 7]; 4];
        let (merged, accepted) =
            merge_expanded_chunks(&circuit, vec![(0, 2, exact.clone())], Some(4));
        assert_eq!(accepted, 1);
        assert_eq!(merged, exact);
    }

    #[test]
    fn mapping_keeps_gate_count_and_stays_within_wire_budget() {
        let order = Permutation {
            data: vec![0, 1, 2],
        };
        let used = vec![7u16, 3, 5];
        let mut rng = rand::rng();
        let mapped = candidate_to_circuit_space(candidate(), false, &order, &used, 10, &mut rng);

        assert_eq!(mapped.gates.len(), 3, "mapping must not add or drop gates");
        for gate in &mapped.gates {
            for &wire in gate {
                assert!(
                    wire < 10,
                    "mapped wires must stay within the circuit budget"
                );
            }
        }
    }

    #[test]
    fn mapping_commutes_with_reversal() {
        // Reversing before relabeling must equal relabeling then reversing: the reversed
        // lookup path only changes gate ORDER, never the wire mapping.
        let order = Permutation {
            data: vec![0, 1, 2],
        };
        let used = vec![7u16, 3, 5];
        let mut rng = rand::rng();
        let forward = candidate_to_circuit_space(candidate(), false, &order, &used, 10, &mut rng);
        let mut reversed =
            candidate_to_circuit_space(candidate(), true, &order, &used, 10, &mut rng);
        reversed.gates.reverse();

        assert_eq!(forward.gates, reversed.gates);
    }

    #[test]
    fn mapping_preserves_internal_collision_structure() {
        // Wire relabeling must not change the relabeling-invariant window features the
        // ranker uses (median leeway, hot-wire touch, window size).
        let order = Permutation {
            data: vec![0, 1, 2],
        };
        let used = vec![7u16, 3, 5];
        let mut rng = rand::rng();
        let mapped = candidate_to_circuit_space(candidate(), false, &order, &used, 10, &mut rng);

        let before = cand_features(&candidate().gates, &[], &[]);
        let after = cand_features(&mapped.gates, &[], &[]);
        assert_eq!(before.size, after.size);
        assert_eq!(before.median_leeway, after.median_leeway);
        assert_eq!(before.max_wire_touch, after.max_wire_touch);
        assert_eq!(before.wires_spanned, after.wires_spanned);
    }
}
