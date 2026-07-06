// Replacement code used in the mixing methods

use crate::replace::mixing::split_into_random_chunk_ranges;
use crate::replace::sat_score::{
    compression_selection_score, expansion_selection_score, sat_bcp_enabled,
    sat_bcp_min_resistance, sat_compress_preserve_delta, sat_compress_protect_enabled,
    sat_cone_aware_enabled, sat_cone_min_fraction, sat_expand_min_delta, sat_score_seed,
    sat_score_slack, sat_scoring_enabled, score_subcircuit,
};
use crate::{
    circuit::circuit::{CircuitSeq, Permutation},
    random::random_data::{
        contiguous_convex, find_convex_subcircuit_max_gates, find_convex_subcircuit_max_wires,
        simple_find_convex_subcircuit,
    },
};
use lmdb::Transaction;
use rand::Rng;
use rayon::iter::IntoParallelIterator;
use rayon::iter::ParallelIterator;
use std::fs::File;
use std::io::{BufWriter, Write};

extern crate lmdb_sys;

use dashmap::DashMap;
use once_cell::sync::Lazy;
use std::sync::Mutex;
use std::sync::OnceLock;
use std::sync::atomic::{AtomicBool, AtomicU64, AtomicUsize, Ordering};
use std::{
    cmp::{max, min},
    time::Instant,
};
// use rand::prelude::IndexedRandom;

// Global histogram: (before_gates, after_gates) -> count, accumulated across all rounds
pub static COMPRESSION_HISTOGRAM: Lazy<DashMap<(u8, u8), u64>> = Lazy::new(DashMap::new);

// Global histograms for EXPANSIONS made in the shuffle-shoot-shuffle game, accumulated
// across all rounds. One is keyed by gate counts, the other by distinct-wire counts.
pub static EXPANSION_HISTOGRAM: Lazy<DashMap<(u8, u8), u64>> = Lazy::new(DashMap::new);
pub static EXPANSION_WIRE_HISTOGRAM: Lazy<DashMap<(u8, u8), u64>> = Lazy::new(DashMap::new);

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

// ---- Per-replacement recording (--record / --rc), option A: momentary positions ----
// When enabled, every expansion/compression replacement appends one line to a log file.
// Positions ("out=start-end") are the gate indices in the working buffer AT THE MOMENT of
// the replacement, NOT stable indices into the final circuit (the pipeline re-splices and
// re-numbers continuously). Each record is tagged with round and a context counter (the
// shooting pass for expansions, the compression iteration for compressions).
pub static RECORD_ENABLED: AtomicBool = AtomicBool::new(false);
pub static REC_ROUND: AtomicUsize = AtomicUsize::new(0); // current round (1-based)
pub static REC_PASS: AtomicUsize = AtomicUsize::new(0); // current shooting pass (expand ctx)
pub static REC_ITER: AtomicUsize = AtomicUsize::new(0); // current compression iteration (compress ctx)
static REC_SEQ: AtomicUsize = AtomicUsize::new(0);
static REC_SINK: Lazy<Mutex<Option<BufWriter<File>>>> = Lazy::new(|| Mutex::new(None));

#[inline]
pub fn record_enabled() -> bool {
    RECORD_ENABLED.load(Ordering::Relaxed)
}

// Open the record log and enable recording. Call once at the start of a run.
pub fn record_init(path: &str) {
    let f = File::create(path).expect("Failed to create replacement record file");
    let mut w = BufWriter::new(f);
    writeln!(
        w,
        "# per-replacement log. out=start-end are gate indices in the working buffer at the moment\n\
         # of the replacement (momentary, not final-circuit indices). ctx = shooting pass (expand)\n\
         # or compression iteration (compress).\n\
         # seq stage round ctx out_start-out_end out_gates in_gates wires"
    )
    .ok();
    *REC_SINK.lock().unwrap() = Some(w);
    RECORD_ENABLED.store(true, Ordering::Relaxed);
}

// Record one replacement. `ctx` is the pass (expand) or iteration (compress) number.
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
    let mut ws: Vec<u16> = wires.to_vec();
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
            out_end - out_start,
            in_gates,
            wires_str
        )
        .ok();
    }
}

// Flush and close the record log at the end of a run.
pub fn record_finish() {
    if let Some(w) = REC_SINK.lock().unwrap().as_mut() {
        w.flush().ok();
    }
}

// ---- Survivor tracking (--track-survivors) ----
// Each gate present right before local mixing starts is tagged with its index (0..N).
// Gates created during mixing carry the sentinel Tag::NEW. The tag vector is maintained in
// lockstep with the gate vector through the shuffle + compression. At the end, the original
// tags still present are the gates that were NEVER part of a replacement (survivors).
//
// Packed per-gate tag. Survivor mode stores the raw origin index (Tag::NEW for new gates),
// so raw equality/comparisons keep working. Gen-mode layout: bits 0..14 = generation
// (saturating at 16383), bits 14..24 = litter_size (saturating at 1023), bits 24..64 =
// litter_id — one fresh id per replacement event, shared by every gate the event creates.
// Gen-mode code must read generations via .generation(), never compare packed values.
#[derive(Copy, Clone, PartialEq, Eq, Debug, Hash)]
pub struct Tag(pub u64);

static NEXT_LITTER_ID: AtomicU64 = AtomicU64::new(0);

impl Tag {
    pub const NEW: Tag = Tag(u64::MAX);

    pub fn new_litter(generation: u32, size: usize) -> Tag {
        let id = NEXT_LITTER_ID.fetch_add(1, Ordering::Relaxed);
        Tag((generation as u64).min(16383) | ((size as u64).min(1023) << 14) | (id << 24))
    }

    pub fn survivor(idx: usize) -> Tag {
        Tag(idx as u64)
    }

    pub fn generation(self) -> u32 {
        (self.0 & 0x3FFF) as u32
    }

    pub fn litter_size(self) -> usize {
        ((self.0 >> 14) & 0x3FF) as usize
    }

    pub fn litter_id(self) -> u64 {
        self.0 >> 24
    }
}

pub static TRACK_SURVIVORS: AtomicBool = AtomicBool::new(false);

#[inline]
pub fn track_survivors() -> bool {
    TRACK_SURVIVORS.load(Ordering::Relaxed)
}

// ---- Generation mode (ssg) ----
// In gen mode the per-gate tag vector holds the gate's GENERATION instead of an origin id:
// feistelized gates are generation 0; the gates a replacement adds get
// floor(median(generations of the removed window)) + 1. (track_survivors() is also true in
// gen mode so the tag vector is maintained; the difference is what new gates are tagged with
// and how the result is reported.)
pub static GEN_MODE: AtomicBool = AtomicBool::new(false);
pub static MAX_FANOUT: AtomicUsize = AtomicUsize::new(50);
pub static MIN_MEDIAN_LEEWAY: AtomicUsize = AtomicUsize::new(10);
// Adaptive plain-SAMF reduction: if a round's shooting game HIDES at least this many SAMFs,
// the explicit plain-SAMF insertion pass is skipped (m set to 0) for subsequent rounds, since
// enough scrambling SAMFs are already woven in. 0 = disabled (always use the fixed m).
pub static SAMF_TARGET: AtomicUsize = AtomicUsize::new(0);

// ---- Stage B: min-generation-anchored bidirectional bounded shooting passes (gen mode) ----
// MIN_GEN: keep launching shooting passes until every gate's generation is >= this.
// PASS_LENGTH: max successful replacements per pass (0 = run to the end of the circuit).
// MAX_PASSES: safety cap on the number of passes per round (prevents non-termination when some
// gates can never be raised, e.g. gates that never collide).
pub static MIN_GEN: AtomicUsize = AtomicUsize::new(1);
pub static PASS_LENGTH: AtomicUsize = AtomicUsize::new(0);
pub static MAX_PASSES: AtomicUsize = AtomicUsize::new(100_000);
// Stop once at least this fraction (in permille, parts per 1000) of gates have generation
// >= MIN_GEN, rather than requiring *every* gate (which converges very slowly because some
// gates rarely collide). Default 990 = 99%.
pub static MIN_GEN_PERMILLE: AtomicUsize = AtomicUsize::new(990);

// ---- Stage D: size-threshold compression cadence (gen mode) ----
// GROW_THRESHOLD_PERMILLE: when > 0, the ssg driver abandons the fixed `-r` round count and
// instead alternates shoot/compress "stages": each stage shoots until the working circuit is
// (1000 + GROW_THRESHOLD_PERMILLE)/1000 times the size it had at the end of the previous
// compression, then compresses all the way back down. The whole cadence stops when the
// min-gen condition (MIN_GEN / MIN_GEN_PERMILLE) is satisfied. 0 = Stage D off (use `-r`).
pub static GROW_THRESHOLD_PERMILLE: AtomicUsize = AtomicUsize::new(0);
// SHOOT_SIZE_CAP: the gen-mode shooting loop pauses (returns to the driver) once the working
// circuit reaches this many gates, so the driver can compress before resuming. Set per stage by
// the Stage D driver; 0 = no cap (shoot until the min-gen target is met, the normal behavior).
pub static SHOOT_SIZE_CAP: AtomicUsize = AtomicUsize::new(0);
// COMPRESS_FRACTION_PERMILLE: in Stage D, each stage compresses only until the circuit is this
// fraction (in permille) of its post-shooting size, instead of all the way down. e.g. 550 with a
// 2x grow threshold => each round nets +10% growth. 0 = compress fully each stage.
pub static COMPRESS_FRACTION_PERMILLE: AtomicUsize = AtomicUsize::new(0);
// FORCED_COLLISIONS: count of shooting steps that fell back to a forced pseudo-collision (the shot
// found no real collision, so the first commuted-past gate was used as a forced collider). Gen mode.
pub static FORCED_COLLISIONS: AtomicUsize = AtomicUsize::new(0);
// TARGET_SIZE: an absolute steady-state size for Stage D. When > 0, each stage shoots until the
// circuit reaches TARGET_SIZE / x (x = COMPRESS_FRACTION_PERMILLE/1000), then compresses back to
// TARGET_SIZE, holding the compressed size fixed at TARGET_SIZE while generations climb to MIN_GEN.
// Overrides GROW_THRESHOLD_PERMILLE's relative cadence. 0 = off (use the grow-threshold cadence).
pub static TARGET_SIZE: AtomicUsize = AtomicUsize::new(0);

#[inline]
pub fn gen_mode() -> bool {
    GEN_MODE.load(Ordering::Relaxed)
}

// VERIFY_DB_HITS (env): when set, every curated-DB replacement is re-checked for functional
// equivalence to the window it replaces, aborting at the exact site on mismatch. Off by default
// (no per-hit cost); used to deterministically localize the feistalize-at-scale equivalence break.
// The env var is read once and cached (0 = unknown, 1 = off, 2 = on).
pub fn verify_db_hits() -> bool {
    static V: AtomicUsize = AtomicUsize::new(0);
    match V.load(Ordering::Relaxed) {
        2 => true,
        1 => false,
        _ => {
            let on = std::env::var("VERIFY_DB_HITS")
                .map(|s| s == "1" || s.eq_ignore_ascii_case("true"))
                .unwrap_or(false);
            V.store(if on { 2 } else { 1 }, Ordering::Relaxed);
            on
        }
    }
}

// COMPRESS_STALL_FRAC (env): aggressive early-stop for compression. When set to a fraction
// f in (0, 1], compress_loop stops as soon as the total reduction over the last
// COMPRESS_STALL_WINDOW sweeps (default 2) falls below f * current_size (floored at the
// legacy 50 gates). Unset = legacy rule: < 50 gates reduced over the last `stable_max` sweeps.
pub fn compress_stall_frac() -> Option<f64> {
    static V: std::sync::OnceLock<Option<f64>> = std::sync::OnceLock::new();
    *V.get_or_init(|| {
        std::env::var("COMPRESS_STALL_FRAC")
            .ok()
            .and_then(|s| s.trim().parse::<f64>().ok())
            .filter(|f| *f > 0.0 && *f <= 1.0)
    })
}

pub fn compress_stall_window() -> usize {
    static V: std::sync::OnceLock<usize> = std::sync::OnceLock::new();
    *V.get_or_init(|| {
        std::env::var("COMPRESS_STALL_WINDOW")
            .ok()
            .and_then(|s| s.trim().parse::<usize>().ok())
            .filter(|w| *w >= 1)
            .unwrap_or(2)
    })
}

// COMPRESS_CHUNK_BUDGET_MS (env): per-chunk wall-clock budget for compression sweeps.
// A sweep only finishes when its slowest chunk does; chunk costs are heavily skewed
// (a few chunks full of slow compress_lmdb windows can serialize a sweep down to one
// core for most of its wall time). When set, compress_big_ancillas stops starting new
// trials once the chunk has run this many ms — the chunk returns its partial progress
// and the re-randomized chunking of the next sweep picks up the region again.
// Unset = legacy (all trials always run).
pub fn compress_chunk_budget_ms() -> Option<u64> {
    static V: std::sync::OnceLock<Option<u64>> = std::sync::OnceLock::new();
    *V.get_or_init(|| {
        std::env::var("COMPRESS_CHUNK_BUDGET_MS")
            .ok()
            .and_then(|s| s.trim().parse::<u64>().ok())
            .filter(|ms| *ms > 0)
    })
}

// LITTER_RULES (env): litter-aware outgoing-window selection. A "litter" is the set of gates
// one replacement event created (they share a Tag litter_id/litter_size). When set, windows
// that consist of exactly one entire litter are banned — removing a full litter is the undo
// of its insertion — and among sampled windows the one spanning the most distinct litters is
// preferred. Shooting-game exception: a full-litter window is allowed if the step lands a
// hidden SAMF (the SAMF is new entropy, so the move still advances the mixing).
// Final compression (COMPRESS_FINAL_PHASE) is exempt. Gen mode only. Unset = legacy.
pub fn litter_rules() -> bool {
    static V: AtomicUsize = AtomicUsize::new(0);
    match V.load(Ordering::Relaxed) {
        2 => true,
        1 => false,
        _ => {
            let on = std::env::var("LITTER_RULES")
                .map(|s| s == "1" || s.eq_ignore_ascii_case("true"))
                .unwrap_or(false);
            V.store(if on { 2 } else { 1 }, Ordering::Relaxed);
            on
        }
    }
}

// SLOW_COMPRESS (env): compress as slowly as possible while never expanding. Equal-size
// ("lateral") DB replacements are accepted, and among the allowed candidates the LARGEST is
// spliced instead of the smallest — the circuit only shrinks when a window's sole equivalents
// are shorter. The compression phase then doubles as a mixing phase; its stop rules become:
// the explicit size target, the min-gen condition (checked every sweep), or a collapse of the
// successful-move rate (SLOW_COMPRESS_MOVE_STALL, default 8 moves over the last 2 sweeps).
// Gen mode only; final compression is exempt. Unset = legacy.
pub fn slow_compress() -> bool {
    static V: AtomicUsize = AtomicUsize::new(0);
    match V.load(Ordering::Relaxed) {
        2 => true,
        1 => false,
        _ => {
            let on = std::env::var("SLOW_COMPRESS")
                .map(|s| s == "1" || s.eq_ignore_ascii_case("true"))
                .unwrap_or(false);
            V.store(if on { 2 } else { 1 }, Ordering::Relaxed);
            on
        }
    }
}

pub fn slow_compress_move_stall() -> usize {
    static V: std::sync::OnceLock<usize> = std::sync::OnceLock::new();
    *V.get_or_init(|| {
        std::env::var("SLOW_COMPRESS_MOVE_STALL")
            .ok()
            .and_then(|s| s.trim().parse::<usize>().ok())
            .unwrap_or(8)
    })
}

// LITTER_WINDOW_SAMPLES (env): windows sampled per compression trial under LITTER_RULES; the
// non-banned window spanning the most distinct litters is the one attempted. Default 4.
pub fn litter_window_samples() -> usize {
    static V: std::sync::OnceLock<usize> = std::sync::OnceLock::new();
    *V.get_or_init(|| {
        std::env::var("LITTER_WINDOW_SAMPLES")
            .ok()
            .and_then(|s| s.trim().parse::<usize>().ok())
            .filter(|k| *k >= 1)
            .unwrap_or(4)
    })
}

// Set by compress_loop for the duration of a call: the delivery ("final") compression, where
// full-litter removal is allowed and SLOW_COMPRESS does not apply.
pub static COMPRESS_FINAL_PHASE: AtomicBool = AtomicBool::new(false);
#[inline]
pub fn compress_final_phase() -> bool {
    COMPRESS_FINAL_PHASE.load(Ordering::Relaxed)
}

// ---- Litter instrumentation (cumulative; reported per sweep as deltas and per phase) ----
pub static LITTER_WIN_CONSIDERED: AtomicUsize = AtomicUsize::new(0); // compression windows sampled
pub static LITTER_BAN_FULL: AtomicUsize = AtomicUsize::new(0); // full-litter windows rejected (compression)
pub static LITTER_TIER1: AtomicUsize = AtomicUsize::new(0); // chosen windows spanning >= 2 litters
pub static LITTER_TIER2: AtomicUsize = AtomicUsize::new(0); // chosen windows: single litter, partial
pub static LITTER_DISTINCT_SUM: AtomicUsize = AtomicUsize::new(0); // sum of distinct litters over tier-1 picks
pub static LITTER_SAMF_LICENSED: AtomicUsize = AtomicUsize::new(0); // shooting: full litter allowed via hidden SAMF
pub static LITTER_ABORT_NO_SAMF: AtomicUsize = AtomicUsize::new(0); // shooting: full litter aborted (no SAMF landed)
pub static LITTER_LATERAL: AtomicUsize = AtomicUsize::new(0); // equal-size compression moves committed
pub static LITTER_SHRINK: AtomicUsize = AtomicUsize::new(0); // strictly-smaller compression moves committed
pub static LITTER_IDENTITY_SKIP: AtomicUsize = AtomicUsize::new(0); // equal-size candidate identical to window
pub static SWEEP_MOVES: AtomicUsize = AtomicUsize::new(0); // successful replacements in the current sweep

pub fn litter_counters() -> [usize; 10] {
    [
        LITTER_WIN_CONSIDERED.load(Ordering::Relaxed),
        LITTER_BAN_FULL.load(Ordering::Relaxed),
        LITTER_TIER1.load(Ordering::Relaxed),
        LITTER_TIER2.load(Ordering::Relaxed),
        LITTER_DISTINCT_SUM.load(Ordering::Relaxed),
        LITTER_SAMF_LICENSED.load(Ordering::Relaxed),
        LITTER_ABORT_NO_SAMF.load(Ordering::Relaxed),
        LITTER_LATERAL.load(Ordering::Relaxed),
        LITTER_SHRINK.load(Ordering::Relaxed),
        LITTER_IDENTITY_SKIP.load(Ordering::Relaxed),
    ]
}

pub fn litter_report(prefix: &str) {
    let c = litter_counters();
    let avg_distinct = if c[2] > 0 { c[4] as f64 / c[2] as f64 } else { 0.0 };
    println!(
        "{} windows={} banned_full={} tier1={} (avg distinct {:.2}) tier2={} samf_licensed={} aborts={} lateral={} shrink={} identity_skips={}",
        prefix, c[0], c[1], c[2], avg_distinct, c[3], c[5], c[6], c[7], c[8], c[9]
    );
}

// FLOAT_SWEEP (env): after every compression sweep (and once on phase entry), float every gate
// to a uniform random position within its "commutable box" — the maximal interval it can slide
// through past non-colliding neighbors. Decorrelates window boundaries from litter boundaries
// (litters are born contiguous and otherwise stay clumped, which the litter counters show as
// full-litter bans concentrating in the local window-finder mode). Tag vector moves in
// lockstep. Unset = legacy (no floating).
pub fn float_sweep_enabled() -> bool {
    static V: AtomicUsize = AtomicUsize::new(0);
    match V.load(Ordering::Relaxed) {
        2 => true,
        1 => false,
        _ => {
            let on = std::env::var("FLOAT_SWEEP")
                .map(|s| s == "1" || s.eq_ignore_ascii_case("true"))
                .unwrap_or(false);
            V.store(if on { 2 } else { 1 }, Ordering::Relaxed);
            on
        }
    }
}

pub static FLOAT_MOVED: AtomicUsize = AtomicUsize::new(0);
pub static FLOAT_DISPLACEMENT: AtomicUsize = AtomicUsize::new(0);

// One comprehensive float pass: left-to-right with rightward floats, then right-to-left with
// leftward floats, so every position is processed exactly once per direction and each gate can
// reach anywhere in its box. Function-preserving: a gate only ever slides past gates it does
// not collide with. Returns (gates moved, total displacement).
pub fn float_all_gates(
    gates: &mut Vec<[u16; 3]>,
    tags: &mut Vec<Tag>,
    rng: &mut impl rand::Rng,
) -> (usize, usize) {
    use crate::circuit::circuit::Gate;
    let track = !tags.is_empty();
    let n = gates.len();
    let mut moved = 0usize;
    let mut disp = 0usize;
    // Rightward pass.
    let mut i = 0usize;
    while i + 1 < n {
        let g = gates[i];
        let mut bound = i;
        while bound + 1 < n && !Gate::collides_index(&gates[bound + 1], &g) {
            bound += 1;
        }
        if bound > i {
            let t = rng.random_range(i..=bound);
            if t != i {
                let gate = gates.remove(i);
                gates.insert(t, gate);
                if track {
                    let tag = tags.remove(i);
                    tags.insert(t, tag);
                }
                moved += 1;
                disp += t - i;
            }
        }
        i += 1;
    }
    // Leftward pass.
    let mut i = n.saturating_sub(1);
    while i > 0 {
        let g = gates[i];
        let mut bound = i;
        while bound > 0 && !Gate::collides_index(&gates[bound - 1], &g) {
            bound -= 1;
        }
        if bound < i {
            let t = rng.random_range(bound..=i);
            if t != i {
                let gate = gates.remove(i);
                gates.insert(t, gate);
                if track {
                    let tag = tags.remove(i);
                    tags.insert(t, tag);
                }
                moved += 1;
                disp += i - t;
            }
        }
        i -= 1;
    }
    FLOAT_MOVED.fetch_add(moved, Ordering::Relaxed);
    FLOAT_DISPLACEMENT.fetch_add(disp, Ordering::Relaxed);
    (moved, disp)
}

// ---- Tag persistence: `<circuit>.tags` sidecars so a resumed run keeps generations/litters ----
// Format: "ssgtags1 <count>\n" then the packed Tag u64s in hex, whitespace-separated.
pub fn write_tags_sidecar(circuit_path: &str, tags: &[Tag]) {
    if tags.is_empty() {
        return;
    }
    let path = format!("{}.tags", circuit_path);
    let mut out = String::with_capacity(tags.len() * 14 + 32);
    out.push_str(&format!("ssgtags1 {}\n", tags.len()));
    for (i, t) in tags.iter().enumerate() {
        out.push_str(&format!("{:x}", t.0));
        out.push(if (i + 1) % 64 == 0 { '\n' } else { ' ' });
    }
    if !out.ends_with('\n') {
        out.push('\n');
    }
    if let Err(e) = std::fs::write(&path, out) {
        eprintln!("[tags] failed to write {}: {}", path, e);
    }
}

pub fn read_tags_sidecar(circuit_path: &str) -> Option<Vec<Tag>> {
    let path = format!("{}.tags", circuit_path);
    let data = std::fs::read_to_string(&path).ok()?;
    let mut it = data.split_ascii_whitespace();
    if it.next()? != "ssgtags1" {
        eprintln!("[tags] {}: unrecognized format, ignoring", path);
        return None;
    }
    let count: usize = it.next()?.parse().ok()?;
    let tags: Option<Vec<Tag>> = it.map(|s| u64::from_str_radix(s, 16).ok().map(Tag)).collect();
    let tags = tags?;
    if tags.len() != count {
        eprintln!("[tags] {}: expected {} tags, found {} — ignoring", path, count, tags.len());
        return None;
    }
    Some(tags)
}

// After loading a sidecar in gen mode, continue minting litter ids above everything loaded.
pub fn bump_litter_ids_past(tags: &[Tag]) {
    let max_id = tags
        .iter()
        .filter(|t| **t != Tag::NEW)
        .map(|t| t.litter_id())
        .max()
        .unwrap_or(0);
    let next = max_id.saturating_add(1);
    if next > NEXT_LITTER_ID.load(Ordering::Relaxed) {
        NEXT_LITTER_ID.store(next, Ordering::Relaxed);
    }
}

// One-line generation + litter distribution report (gen mode).
pub fn gen_report(prefix: &str, tags: &[Tag]) {
    if tags.is_empty() {
        return;
    }
    let mut gens: Vec<u32> = tags.iter().map(|t| t.generation()).collect();
    gens.sort_unstable();
    let len = gens.len();
    let permille = MIN_GEN_PERMILLE.load(Ordering::Relaxed).min(1000);
    let floor_q = gens[(len * (1000 - permille)) / 1000];
    let median = gens[len / 2];
    let max = gens[len - 1];
    let mg = MIN_GEN.load(Ordering::Relaxed);
    let at_target = len - gens.partition_point(|g| (*g as usize) < mg);
    let mut ids: Vec<u64> = tags.iter().map(|t| t.litter_id()).collect();
    ids.sort_unstable();
    ids.dedup();
    println!(
        "{} floor{}={} median={} max={} | >=gen{}: {:.1}% | litters={} (avg size {:.1})",
        prefix,
        permille / 10,
        floor_q,
        median,
        max,
        mg,
        100.0 * at_target as f64 / len as f64,
        ids.len(),
        len as f64 / ids.len().max(1) as f64
    );
}

// Litter profile of a candidate outgoing window: (#distinct litters, is-exactly-one-full-litter).
// Only meaningful in gen mode with tag tracking on.
pub fn window_litter_stats(tags: &[Tag]) -> (usize, bool) {
    if tags.is_empty() {
        return (0, false);
    }
    let first = tags[0];
    if tags.iter().all(|t| *t == first) {
        return (1, tags.len() == first.litter_size());
    }
    let mut ids: Vec<u64> = tags.iter().map(|t| t.litter_id()).collect();
    ids.sort_unstable();
    ids.dedup();
    (ids.len(), false)
}

// DEGREE_FILTER (env): d = the max ANF degree any curated-DB function can have. Measured
// distribution: main-DB classes top out at degree 7 (= the k+1 bound for 6-gate circuits;
// a k-gate circuit has ANF degree <= k+1, tight), so a compression window whose function has
// certified degree > d is an unmatchable miss and canonicalization can be skipped entirely.
// Default d = 8 (one point of margin over the observed 7). Unset = off.
pub fn degree_filter() -> Option<usize> {
    static V: std::sync::OnceLock<Option<usize>> = std::sync::OnceLock::new();
    *V.get_or_init(|| {
        std::env::var("DEGREE_FILTER")
            .ok()
            .and_then(|s| s.trim().parse::<usize>().ok())
            .filter(|d| *d >= 1)
    })
}

pub fn degree_filter_probes() -> usize {
    static V: std::sync::OnceLock<usize> = std::sync::OnceLock::new();
    *V.get_or_init(|| {
        std::env::var("DEGREE_FILTER_PROBES")
            .ok()
            .and_then(|s| s.trim().parse::<usize>().ok())
            .filter(|k| *k >= 1)
            .unwrap_or(6)
    })
}

pub static DEGREE_FILTER_SKIPS: AtomicUsize = AtomicUsize::new(0);

// Bit-sliced witness for "degree > d": pick d+1 active input wires, fix the rest to random
// constants, and XOR the given output wire's value over the 2^(d+1) active-subcube. By the
// Mobius relation that XOR-sum equals the ANF coefficient of the full active monomial in the
// restricted function; a 1 certifies degree >= d+1 for the whole function (restriction never
// raises degree). One-sided: a low-degree function NEVER produces a witness. `gates` are in
// dense wire space (0..nw). Returns true iff some output wire's subcube parity is 1.
// Evaluate the (d+1)-th order derivative of the window along `dirs` (m = d+1 random direction
// vectors over the nw wires) at random base point `base`, i.e. XOR the outputs over the affine
// subcube { base ⊕ Σ εᵢ·dirsᵢ : ε ∈ {0,1}^m }. That XOR-sum equals D_{dirs} f (the iterated
// discrete derivative), which is identically 0 for EVERY choice when deg f ≤ d and nonzero for
// some choice when deg f ≥ d+1 — and, unlike an axis-aligned subcube, random directions catch a
// degree-D monomial regardless of which wires carry it. One-sided: a low-degree function never
// produces a witness. `gates` in dense wire space (0..nw). Returns true iff some output wire's
// derivative is 1. `dirs[i][w]` = does direction i include wire w; `base[w]` = base bit for w.
fn derivative_witness(
    gates: &[[u16; 3]],
    nw: usize,
    dirs: &[Vec<bool>],
    base: &[bool],
) -> bool {
    let m = dirs.len();
    let total = 1usize << m; // affine-subcube points
    let words = total.div_ceil(64);
    // Axis columns: col[i] has bit p set iff (p>>i)&1 == 1 (the value of εᵢ at point p).
    let mut col = vec![vec![0u64; words]; m];
    for (i, ci) in col.iter_mut().enumerate() {
        for p in 0..total {
            if (p >> i) & 1 == 1 {
                ci[p / 64] |= 1u64 << (p % 64);
            }
        }
    }
    // Wire w value across points = base[w] (constant) XOR of the columns of directions hitting w.
    let mut state = vec![vec![0u64; words]; nw];
    for w in 0..nw {
        if base[w] {
            for x in state[w].iter_mut() {
                *x = u64::MAX;
            }
        }
        for (i, di) in dirs.iter().enumerate() {
            if di[w] {
                for wi in 0..words {
                    state[w][wi] ^= col[i][wi];
                }
            }
        }
    }
    // g57 update, matching to_polynomial exactly: a' = a XOR NOT(b AND NOT c) (= a XOR 1 XOR b XOR bc).
    for &[a, b, c] in gates {
        let (a, b, c) = (a as usize, b as usize, c as usize);
        for wi in 0..words {
            let v = !(state[b][wi] & !state[c][wi]);
            state[a][wi] ^= v;
        }
    }
    let mask_last = if total % 64 == 0 {
        u64::MAX
    } else {
        (1u64 << (total % 64)) - 1
    };
    for w in 0..nw {
        let mut par = 0u32;
        for wi in 0..words {
            let word = if wi + 1 == words {
                state[w][wi] & mask_last
            } else {
                state[w][wi]
            };
            par ^= word.count_ones() & 1;
        }
        if par & 1 == 1 {
            return true;
        }
    }
    false
}

// Certify that the window's function IN ONE DIRECTION has degree > d (reversed = the inverse
// permutation, since g57 gates are involutions so the inverse is the reversed gate list). One-
// sided: returns true only on a witness (never a false positive), so a `true` means the DB
// lookup for that direction is a guaranteed miss and its canonicalization can be skipped.
pub fn degree_exceeds_dir(
    sub: &CircuitSeq,
    reversed: bool,
    d: usize,
    k: usize,
    rng: &mut impl rand::Rng,
) -> bool {
    let used = sub.used_wires();
    let nw = used.len();
    if nw <= d || d + 1 > 12 {
        return false; // degree <= #inputs <= d, or subcube too large to probe cheaply
    }
    let wire_map: std::collections::HashMap<u16, u16> =
        used.iter().enumerate().map(|(i, &w)| (w, i as u16)).collect();
    let mut dense: Vec<[u16; 3]> = sub
        .gates
        .iter()
        .map(|&[t, c1, c2]| [wire_map[&t], wire_map[&c1], wire_map[&c2]])
        .collect();
    if reversed {
        dense.reverse();
    }
    for _ in 0..k {
        // d+1 random nonzero direction vectors over the nw wires, plus a random base point.
        let dirs: Vec<Vec<bool>> = (0..d + 1)
            .map(|_| {
                let mut v: Vec<bool> = (0..nw).map(|_| rng.random_bool(0.5)).collect();
                if v.iter().all(|&b| !b) {
                    v[rng.random_range(0..nw)] = true; // avoid the zero direction
                }
                v
            })
            .collect();
        let base: Vec<bool> = (0..nw).map(|_| rng.random_bool(0.5)).collect();
        if derivative_witness(&dense, nw, &dirs, &base) {
            return true;
        }
    }
    false
}

// Should the whole compression trial be discarded? Only when BOTH directions certify degree > d
// (a window whose function has deg > d but whose INVERSE has deg <= d is still compressible via
// the reverse DB lookup, so it must NOT be discarded). Kept for tests; the live code guards each
// canonicalization direction separately via degree_exceeds_dir so the forward explosion is
// avoided even when only the forward direction is high.
pub fn degree_filter_discards(sub: &CircuitSeq, d: usize, k: usize, rng: &mut impl rand::Rng) -> bool {
    if !degree_exceeds_dir(sub, false, d, k, rng) {
        return false; // forward lookup might hit
    }
    degree_exceeds_dir(sub, true, d, k, rng)
}

// Floor of the median of the window's generations (round the even-length midpoint down).
pub fn median_floor_gen(v: &[Tag]) -> u32 {
    if v.is_empty() {
        return 0;
    }
    let mut s: Vec<u32> = v.iter().map(|t| t.generation()).collect();
    s.sort_unstable();
    let n = s.len();
    if n % 2 == 1 {
        s[n / 2]
    } else {
        ((s[n / 2 - 1] as u64 + s[n / 2] as u64) / 2) as u32
    }
}

// Tag assigned to the `new_count` gates a replacement ADDS, given the removed window's tags.
// gen mode -> a fresh litter at floor(median(window)) + 1 ; survivor mode -> Tag::NEW.
#[inline]
pub fn new_gate_tag(window_tags: &[Tag], new_count: usize) -> Tag {
    if gen_mode() {
        Tag::new_litter(median_floor_gen(window_tags).saturating_add(1), new_count)
    } else {
        Tag::NEW
    }
}

// ---- Stage E: fanout/leeway-driven replacement selection (gen mode) ----
// Target fanout distribution: fractions of gates with fanout 0,1,2,3,>3.
pub const FANOUT_TARGET: [f64; 5] = [0.25, 0.40, 0.20, 0.10, 0.05];

// Fanout of each gate = number of later gates that read its target wire (as a control)
// before that wire is next written.
pub fn gate_fanouts(gates: &[[u16; 3]]) -> Vec<usize> {
    let n = gates.len();
    let mut out = vec![0usize; n];
    for i in 0..n {
        let w = gates[i][0];
        for g in &gates[i + 1..] {
            if g[0] == w {
                break; // target rewritten -> later reads belong to the new writer
            }
            if g[1] == w || g[2] == w {
                out[i] += 1;
            }
        }
    }
    out
}

#[inline]
fn fanout_bucket(f: usize) -> usize {
    f.min(4)
}

// L1 distance of a gate list's fanout-bucket distribution to FANOUT_TARGET (lower = better).
pub fn fanout_target_l1(gates: &[[u16; 3]]) -> f64 {
    if gates.is_empty() {
        return f64::INFINITY;
    }
    let mut b = [0usize; 5];
    for f in gate_fanouts(gates) {
        b[fanout_bucket(f)] += 1;
    }
    let n = gates.len() as f64;
    (0..5)
        .map(|k| (b[k] as f64 / n - FANOUT_TARGET[k]).abs())
        .sum()
}

pub fn max_fanout_exceeded(gates: &[[u16; 3]]) -> bool {
    let cap = MAX_FANOUT.load(Ordering::Relaxed);
    gate_fanouts(gates).into_iter().any(|f| f > cap)
}

// Leeway of each gate = how far it can commute left + right within the list before colliding.
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

// Number of gates whose leeway is below MIN_MEDIAN_LEEWAY (lower = better when raising leeway).
pub fn low_leeway_count(gates: &[[u16; 3]]) -> usize {
    let thr = MIN_MEDIAN_LEEWAY.load(Ordering::Relaxed);
    gate_leeways(gates).into_iter().filter(|&lw| lw < thr).count()
}

// Global-context features of a candidate `window` placed between `left` and `right` context.
// Fanout of a gate = readers of its target wire until that wire is next written (bounded scan
// into the window tail then `right`). Leeway = how far it commutes left+right (into the window
// then `left`/`right`) before colliding. wires_spanned = distinct wires in the window.
pub fn cand_features(
    window: &[[u16; 3]],
    left: &[[u16; 3]],
    right: &[[u16; 3]],
) -> crate::replace::ranking::CandFeatures {
    use crate::circuit::circuit::Gate;
    use crate::replace::ranking::CandFeatures;
    let thr = MIN_MEDIAN_LEEWAY.load(Ordering::Relaxed);
    let n = window.len();
    let mut wires = std::collections::HashSet::new();
    for g in window {
        wires.insert(g[0]);
        wires.insert(g[1]);
        wires.insert(g[2]);
    }
    let mut buckets = [0usize; 5];
    let (mut zero_fanout, mut low_leeway, mut maxf) = (0usize, 0usize, 0usize);
    for i in 0..n {
        let w = window[i][0];
        // fanout: readers of w after i (window tail, then right) until w is rewritten
        let mut f = 0usize;
        let mut rewritten = false;
        for g in &window[i + 1..] {
            if g[0] == w {
                rewritten = true;
                break;
            }
            if g[1] == w || g[2] == w {
                f += 1;
            }
        }
        if !rewritten {
            for g in right {
                if g[0] == w {
                    break;
                }
                if g[1] == w || g[2] == w {
                    f += 1;
                }
            }
        }
        // leeway: commute left then right until collision
        let cur = &window[i];
        let mut lee = 0usize;
        let mut blocked = false;
        for g in window[..i].iter().rev() {
            if Gate::collides_index(g, cur) {
                blocked = true;
                break;
            }
            lee += 1;
        }
        if !blocked {
            for g in left.iter().rev() {
                if Gate::collides_index(g, cur) {
                    break;
                }
                lee += 1;
            }
        }
        blocked = false;
        for g in &window[i + 1..] {
            if Gate::collides_index(g, cur) {
                blocked = true;
                break;
            }
            lee += 1;
        }
        if !blocked {
            for g in right {
                if Gate::collides_index(g, cur) {
                    break;
                }
                lee += 1;
            }
        }
        buckets[f.min(4)] += 1;
        if f == 0 {
            zero_fanout += 1;
        }
        if lee < thr {
            low_leeway += 1;
        }
        if f > maxf {
            maxf = f;
        }
    }
    CandFeatures {
        size: n,
        wires_spanned: wires.len(),
        low_leeway_count: low_leeway,
        zero_fanout_count: zero_fanout,
        fanout_buckets: buckets,
        max_fanout: maxf,
    }
}

// Pick the best candidates by (fanout L1 to target, then low-leeway count), after applying the
// MAX_FANOUT hard cap. Returns the tied-best set (caller breaks ties randomly).
pub fn gen_select_best(candidates: Vec<CircuitSeq>) -> Vec<CircuitSeq> {
    if candidates.is_empty() {
        return candidates;
    }
    // Hard cap on fanout; if every candidate violates, fall back to all (best effort).
    let kept: Vec<CircuitSeq> = candidates
        .iter()
        .filter(|c| !max_fanout_exceeded(&c.gates))
        .cloned()
        .collect();
    let pool = if kept.is_empty() { candidates } else { kept };
    let scored: Vec<(f64, usize, CircuitSeq)> = pool
        .into_iter()
        .map(|c| (fanout_target_l1(&c.gates), low_leeway_count(&c.gates), c))
        .collect();
    let best = scored
        .iter()
        .map(|(l1, ll, _)| (*l1, *ll))
        .min_by(|a, b| a.0.partial_cmp(&b.0).unwrap().then(a.1.cmp(&b.1)))
        .unwrap();
    scored
        .into_iter()
        .filter(|(l1, ll, _)| (*l1 - best.0).abs() < 1e-9 && *ll == best.1)
        .map(|(_, _, c)| c)
        .collect()
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
pub static TXN_TIME: AtomicU64 = AtomicU64::new(0);
pub static LMDB_LOOKUP_TIME: AtomicU64 = AtomicU64::new(0);
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

pub fn compress_loop(
    circuit: &CircuitSeq,
    n: usize,
    env: &lmdb::Environment,
    shard_dbs: &[lmdb::Database],
    stable_max: usize,
    curr_round: usize,
    last_round: usize,
    output_path: &str,
    light_compression: bool,
    // When `Some(t)`, stop early as soon as the circuit is reduced to <= `t` gates (used by the
    // standalone `compress` command to stop at a fraction of the initial size). Independent of the
    // no-progress (`stable_max`) and `light_compression` stops; whichever triggers first wins.
    early_stop_target: Option<usize>,
    // Delivery compression: full-litter removal allowed, SLOW_COMPRESS off, legacy stop rules.
    final_phase: bool,
    tags: &mut Vec<Tag>,
) -> CircuitSeq {
    COMPRESS_FINAL_PHASE.store(final_phase, Ordering::Relaxed);
    let track = !tags.is_empty();
    let mut acc = circuit.clone();
    // Origin tags maintained in lockstep with `acc.gates` (for --track-survivors).
    let mut acc_tags: Vec<Tag> = tags.clone();
    // --light-compression: stop once the circuit is reduced to at most half the
    // max (post-shooting) size, i.e. the size on entry to this compress_loop call.
    let max_size = circuit.gates.len();
    let light_target = max_size / 2;
    let mut rng = rand::rng();
    let mut mode = 0usize;
    // Ring buffer of the last stable_max+1 gate counts. Stop when total reduction
    // over the last stable_max iterations is less than 100 gates.
    let mut recent: std::collections::VecDeque<usize> =
        std::collections::VecDeque::with_capacity(stable_max + 1);
    recent.push_back(acc.gates.len());
    let slow_active = gen_mode() && slow_compress() && !final_phase && track;
    let litter_active = gen_mode() && litter_rules() && !final_phase && track;
    // Successful-move counts of recent sweeps (slow-mode stall is on moves, not gates shaved).
    let mut recent_moves: std::collections::VecDeque<usize> =
        std::collections::VecDeque::with_capacity(3);
    let mut prev_litter = litter_counters();
    if slow_active {
        println!(
            "  [compress] slow mode: lateral moves on, largest-allowed pick, move-stall < {} over 2 sweeps",
            slow_compress_move_stall()
        );
    }
    if litter_active {
        println!(
            "  [litter] rules active: full-litter windows banned, {} samples per trial",
            litter_window_samples()
        );
    }
    if !slow_active {
        if let Some(f) = compress_stall_frac() {
            println!(
                "  [compress] stall rule: stop when < {:.1}% of current size reduced over last {} sweeps",
                f * 100.0,
                compress_stall_window().min(stable_max)
            );
        }
    }
    if let Some(ms) = compress_chunk_budget_ms() {
        println!("  [compress] chunk budget: {} ms per chunk per sweep", ms);
    }

    let mut rec_iter = 0usize;
    loop {
        let before = acc.gates.len();
        rec_iter += 1;
        REC_ITER.store(rec_iter, Ordering::Relaxed);
        SWEEP_MOVES.store(0, Ordering::Relaxed);

        // FLOAT_SWEEP: rerandomize every gate's position within its commutable box before the
        // sweep (so this runs on phase entry and again after each completed sweep).
        if float_sweep_enabled() && !final_phase {
            let (fm, fd) = float_all_gates(&mut acc.gates, &mut acc_tags, &mut rng);
            println!(
                "  [float] sweep {}: moved {} of {} gates, avg displacement {:.1}",
                rec_iter,
                fm,
                acc.gates.len(),
                if fm > 0 { fd as f64 / fm as f64 } else { 0.0 }
            );
        }

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
        let compressed_chunks: Vec<(usize, usize, usize, Vec<[u16; 3]>, Vec<Tag>, u128)> = ranges
            .into_iter()
            .enumerate()
            .collect::<Vec<_>>()
            .into_par_iter()
            .map(|(chunk_idx, (start, end))| {
                let sub = CircuitSeq {
                    gates: acc.gates[start..end].to_vec(),
                };
                // Per-chunk tags; base_offset=start makes recorded positions circuit-relative.
                let mut chunk_tags: Vec<Tag> = if track {
                    acc_tags_ref[start..end].to_vec()
                } else {
                    Vec::new()
                };
                let chunk_start = Instant::now();
                let gates =
                    compress_big_ancillas(&sub, 100, n, env, shard_dbs, current_mode, start, &mut chunk_tags)
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
            let total_chunk_ms: u128 = compressed_chunks.iter().map(|(_, _, _, _, _, ms)| *ms).sum();
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
        let mut new_tags: Vec<Tag> = if track {
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
        if recent.len() > stable_max + 1 {
            recent.pop_front();
        }
        let sweep_moves = SWEEP_MOVES.load(Ordering::Relaxed);

        if litter_active || slow_active {
            let cur = litter_counters();
            let d: Vec<usize> = cur.iter().zip(prev_litter.iter()).map(|(a, b)| a - b).collect();
            println!(
                "  [litter] sweep {} (mode {}): moves={} (lateral {} / shrink {}) banned_full={} tier1={} tier2={} identity_skips={}",
                rec_iter, current_mode, sweep_moves, d[7], d[8], d[1], d[2], d[3], d[9]
            );
            prev_litter = cur;
        }
        if track && gen_mode() {
            gen_report(
                &format!("  [gen] sweep {} (mode {}):", rec_iter, current_mode),
                &acc_tags,
            );
        }

        if slow_active {
            // Slow-mode stops, in priority order: (1) min-gen condition met — the phase mixed
            // the circuit to completion, exit straight toward delivery; (2) the explicit size
            // target (below, shared with legacy); (3) move-rate collapse — nothing left to do.
            // The legacy gates-shaved stall is meaningless here (lateral moves shave nothing).
            let mg = MIN_GEN.load(Ordering::Relaxed);
            let need_permille = MIN_GEN_PERMILLE.load(Ordering::Relaxed);
            let total = acc_tags.len().max(1);
            let met = acc_tags
                .iter()
                .filter(|t| (t.generation() as usize) >= mg)
                .count();
            if met * 1000 >= need_permille * total {
                println!(
                    "  {}/{}: [compress] slow-mode stop: min-gen met ({}/{} gates at gen >= {}), {} gates",
                    curr_round, last_round, met, total, mg, after
                );
                break;
            }
            recent_moves.push_back(sweep_moves);
            if recent_moves.len() > 2 {
                recent_moves.pop_front();
            }
            if recent_moves.len() == 2
                && recent_moves.iter().sum::<usize>() < slow_compress_move_stall()
            {
                println!(
                    "  {}/{}: [compress] slow-mode stop: move-stall ({} moves over last 2 sweeps, threshold {}), {} gates",
                    curr_round,
                    last_round,
                    recent_moves.iter().sum::<usize>(),
                    slow_compress_move_stall(),
                    after
                );
                break;
            }
        } else {
            // Stall stop. Legacy: < 50 gates reduced over the last stable_max sweeps.
            // COMPRESS_STALL_FRAC set: < frac * current_size reduced over the last
            // COMPRESS_STALL_WINDOW sweeps (see compress_stall_frac) — fires much earlier.
            let (stall_window, stall_threshold) = match compress_stall_frac() {
                Some(f) => (
                    compress_stall_window().min(stable_max),
                    ((after as f64 * f) as usize).max(50),
                ),
                None => (stable_max, 50),
            };
            if recent.len() > stall_window {
                let base = recent[recent.len() - 1 - stall_window];
                let window_reduction = base.saturating_sub(after);
                if window_reduction < stall_threshold {
                    println!(
                        "  {}/{}: Early stop — only {} gates reduced over last {} iterations, threshold {} ({} gates)",
                        curr_round, last_round, window_reduction, stall_window, stall_threshold, after
                    );
                    break;
                }
            }
        }

        if after == before {
            println!("  {}/{}: Stable ({} gates)", curr_round, last_round, after);
        } else {
            println!("  {}/{}: Reduced: {} gates", curr_round, last_round, after);
        }

        // --light-compression: stop as soon as we are at or below half the max size.
        if light_compression && after <= light_target {
            println!(
                "  {}/{}: Light compression target reached ({} <= {} = max/2 of {}), stopping",
                curr_round, last_round, after, light_target, max_size
            );
            break;
        }

        // Explicit early-stop target (e.g. a fraction of the initial size): stop once reached.
        if let Some(target) = early_stop_target {
            if after <= target {
                println!(
                    "  {}/{}: Early-stop target reached ({} <= {} gates), stopping",
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
            if track {
                write_tags_sidecar(output_path, &acc_tags);
            }
            eprintln!("Wrote {}", output_path);
        }
    }
    if litter_active || slow_active {
        litter_report("  [litter] phase totals:");
    }
    if track {
        *tags = acc_tags;
    }
    acc
}

/// Single pass of expansion: one round of chunked `expand_big_ancillas` with no loop.
pub fn expand_once<'a>(
    circuit: &CircuitSeq,
    n: usize,
    env: &lmdb::Environment,
    shard_dbs: &[lmdb::Database],
    pair_mode: &ExpandPairMode<'a>,
) -> CircuitSeq {
    let mut rng = rand::rng();
    let before = circuit.gates.len();
    let max_chunks = 4 * rayon::current_num_threads().max(1);
    let k = if before <= 1500 {
        1
    } else {
        ((before + 1499) / 1500).min(max_chunks)
    };
    let ranges = split_into_random_chunk_ranges(before, k, &mut rng);
    let expanded_chunks: Vec<Vec<[u16; 3]>> = ranges
        .into_par_iter()
        .map(|(start, end)| {
            let sub = CircuitSeq {
                gates: circuit.gates[start..end].to_vec(),
            };
            expand_big_ancillas(&sub, 100, n, env, shard_dbs, 0, pair_mode).gates
        })
        .collect();
    let mut new_gates = Vec::with_capacity(expanded_chunks.iter().map(|c| c.len()).sum());
    for chunk in expanded_chunks {
        new_gates.extend(chunk);
    }
    println!("  Expand: {} gates", new_gates.len());
    CircuitSeq { gates: new_gates }
}

// Expand with ancilla wires or gates
/// Selects which method to use when a 2-gate subcircuit is sampled in the expand functions.
/// For subcircuits of 3–5 gates the shard DB is always used regardless of this setting.
pub enum ExpandPairMode<'a> {
    /// Use the curated shard DBs to find a longer equivalent pair.
    Curated {
        curated_shard_dbs: &'a [lmdb::Database],
    },
    /// Force the shard DB lookup even for 2-gate subcircuits (same path as 3-5 gates).
    Db,
}

pub fn expand_lmdb<'a>(
    c: &CircuitSeq,
    trials: usize,
    n: usize,
    env: &lmdb::Environment,
    shard_dbs: &[lmdb::Database],
    pair_mode: &ExpandPairMode<'a>,
) -> CircuitSeq {
    use crate::circuit::circuit::polys_repr_blob;
    use xxhash_rust::xxh3::xxh3_128;

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

        // --- 2-gate path: bypass the shard DB and use pair functions ---
        if sub.gates.len() == 2 {
            match pair_mode {
                ExpandPairMode::Curated { curated_shard_dbs } => {
                    use crate::replace::pairs::expand_curated_lmdb;
                    if let Some(repl) =
                        expand_curated_lmdb(&sub.gates, n, env, curated_shard_dbs, shard_dbs)
                    {
                        if repl.len() > 2 {
                            expanded.gates.splice(start..end, repl);
                        }
                    }
                    TRIAL_TIME.fetch_add(t_trial.elapsed().as_nanos() as u64, Ordering::Relaxed);
                    continue;
                }
                ExpandPairMode::Db => { /* fall through to shard DB path */ }
            }
        }

        // Degree pre-filter (per direction): a window function of degree > d cannot equal any
        // curated-DB function (DB circuits have degree <= gates+1 <= 8), so that direction's
        // lookup is a guaranteed miss and its canonicalization — with its monomial-explosion
        // risk — is skipped. The forward and reverse directions are the window's function and its
        // inverse (g57 gates are involutions), which can differ in degree; only when BOTH are
        // high is the whole trial a certain miss.
        let deg_d = degree_filter();
        let fwd_high = deg_d
            .map(|d| degree_exceeds_dir(&sub, false, d, degree_filter_probes(), &mut rng))
            .unwrap_or(false);

        let txn = match env.begin_ro_txn() {
            Ok(t) => t,
            Err(_) => continue,
        };

        // (value, final_order, used, is_reversed)
        let mut matched: Option<(Vec<u8>, Permutation, Vec<u16>, bool)> = None;
        if !fwd_high {
            let t_canon = Instant::now();
            let (fwd_polys, fwd_order, used) = sub.canonicalize_polys_single(false);
            CANONICALIZE_TIME.fetch_add(t_canon.elapsed().as_nanos() as u64, Ordering::Relaxed);
            if !fwd_polys.is_empty() {
                let fwd_key = xxh3_128(&polys_repr_blob(&fwd_polys)).to_le_bytes().to_vec();
                let t_lookup = Instant::now();
                let fwd_result = txn
                    .get(shard_dbs[fwd_key[0] as usize], &fwd_key)
                    .map(|v: &[u8]| v.to_vec());
                LMDB_LOOKUP_TIME.fetch_add(t_lookup.elapsed().as_nanos() as u64, Ordering::Relaxed);
                if let Ok(v) = fwd_result {
                    matched = Some((v, fwd_order, used, false));
                }
            }
        }
        if matched.is_none() {
            let rev_high = deg_d
                .map(|d| degree_exceeds_dir(&sub, true, d, degree_filter_probes(), &mut rng))
                .unwrap_or(false);
            if rev_high {
                if fwd_high {
                    // both directions certified high -> definite miss, canonicalization skipped
                    DEGREE_FILTER_SKIPS.fetch_add(1, Ordering::Relaxed);
                }
            } else {
                let t_canon2 = Instant::now();
                let (rev_polys, rev_order, used) = sub.canonicalize_polys_single(true);
                CANONICALIZE_TIME.fetch_add(t_canon2.elapsed().as_nanos() as u64, Ordering::Relaxed);
                if !rev_polys.is_empty() {
                    let rev_key = xxh3_128(&polys_repr_blob(&rev_polys)).to_le_bytes().to_vec();
                    let t_lookup2 = Instant::now();
                    let rev_result = txn
                        .get(shard_dbs[rev_key[0] as usize], &rev_key)
                        .map(|v: &[u8]| v.to_vec());
                    LMDB_LOOKUP_TIME.fetch_add(t_lookup2.elapsed().as_nanos() as u64, Ordering::Relaxed);
                    if let Ok(v) = rev_result {
                        matched = Some((v, rev_order, used, true));
                    }
                }
            }
        }
        let (value, final_order, used, is_reversed) = match matched {
            Some(x) => x,
            None => continue,
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
            if candidate.gates.len() > sub.gates.len() {
                candidates.push(candidate);
            }
        }
        FROM_BLOB_TIME.fetch_add(t_blob.elapsed().as_nanos() as u64, Ordering::Relaxed);

        if candidates.is_empty() {
            continue;
        }

        let mut best: Vec<CircuitSeq> = if sat_scoring_enabled() {
            let seed = sat_score_seed();
            let base_n = sub.max_wire() + 1;
            let base_score = expansion_selection_score(&score_subcircuit(
                &sub.gates,
                base_n,
                seed ^ 0xE4A5_10AD,
            ));
            let required_score = base_score + sat_expand_min_delta();
            let scored: Vec<(f64, CircuitSeq)> = candidates
                .into_iter()
                .enumerate()
                .filter_map(|(idx, candidate)| {
                    let score_n = candidate.max_wire() + 1;
                    let sat_score = score_subcircuit(&candidate.gates, score_n, seed ^ idx as u64);
                    if sat_cone_aware_enabled()
                        && sat_score.output_cone_fraction < sat_cone_min_fraction()
                    {
                        return None;
                    }
                    if sat_bcp_enabled() && sat_score.bcp_resistance < sat_bcp_min_resistance() {
                        return None;
                    }
                    Some((expansion_selection_score(&sat_score), candidate))
                })
                .filter(|(score, _)| *score > required_score)
                .collect();
            if scored.is_empty() {
                continue;
            }
            let max_score = scored
                .iter()
                .map(|(score, _)| *score)
                .fold(f64::NEG_INFINITY, f64::max);
            scored
                .into_iter()
                .filter(|(score, _)| (*score - max_score).abs() <= 1e-9)
                .map(|(_, candidate)| candidate)
                .collect()
        } else {
            let max_gates = candidates.iter().map(|c| c.gates.len()).max().unwrap();
            candidates
                .into_iter()
                .filter(|c| c.gates.len() == max_gates)
                .collect()
        };
        let idx = rng.random_range(0..best.len());
        let mut repl = best.swap_remove(idx);

        if is_reversed {
            repl.gates.reverse();
        }

        let t_rewire = Instant::now();
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
        let mut used_ext = used.clone();
        if used_ext.len() < repl_n_b {
            let mut available: Vec<u16> = (0..n as u16).filter(|w| !used_ext.contains(w)).collect();
            rand::seq::SliceRandom::shuffle(available.as_mut_slice(), &mut rng);
            let mut avail = available.into_iter();
            while used_ext.len() < repl_n_b {
                match avail.next() {
                    Some(w) => used_ext.push(w),
                    None => break,
                }
            }
        }
        let repl = CircuitSeq::unrewire_subcircuit(&repl, &used_ext);
        REWIRE_TIME.fetch_add(t_rewire.elapsed().as_nanos() as u64, Ordering::Relaxed);

        let t_splice = Instant::now();
        expanded.gates.splice(start..end, repl.gates);
        SPLICE_TIME.fetch_add(t_splice.elapsed().as_nanos() as u64, Ordering::Relaxed);

        TRIAL_TIME.fetch_add(t_trial.elapsed().as_nanos() as u64, Ordering::Relaxed);
    }

    expanded
}

// Map a raw DB candidate (canonical wire space, forward orientation) back into the actual
// circuit wire space: undo reversal, apply the canonicalization order, then assign concrete
// wires (filling any extra ancillas with random unused wires). Returns the circuit-space repl.
fn rewire_candidate(
    mut repl: CircuitSeq,
    is_reversed: bool,
    final_order: &Permutation,
    used: &[u16],
    n: usize,
    rng: &mut impl rand::Rng,
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
        let mut available: Vec<u16> = (0..n as u16).filter(|w| !used_ext.contains(w)).collect();
        rand::seq::SliceRandom::shuffle(available.as_mut_slice(), rng);
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

pub fn compress_lmdb(
    c: &CircuitSeq,
    trials: usize,
    n: usize,
    env: &lmdb::Environment,
    shard_dbs: &[lmdb::Database],
    mode: usize,
    base_offset: usize,
    tags: &mut Vec<Tag>,
) -> CircuitSeq {
    use crate::circuit::circuit::polys_repr_blob;
    use xxhash_rust::xxh3::xxh3_128;

    // `tags` (non-empty when --track-survivors): per-gate origin id, kept in lockstep with
    // `compressed.gates`. `base_offset` makes the recorded replacement position circuit-relative.
    let track = !tags.is_empty();
    let mut compressed = c.clone();

    let mut i = 0;
    while i < compressed.gates.len().saturating_sub(1) {
        if compressed.gates[i] == compressed.gates[i + 1] {
            compressed.gates.drain(i..=i + 1);
            if track {
                tags.drain(i..=i + 1);
            }
            i = i.saturating_sub(2);
        } else {
            i += 1;
        }
    }

    if compressed.gates.is_empty() {
        return CircuitSeq { gates: Vec::new() };
    }

    let mut rng = rand::rng();

    // Litter rules (gen mode, non-final): sample several windows per trial, ban any that is
    // exactly one whole litter (its removal would be the undo of its insertion), and attempt
    // the window spanning the most distinct litters.
    let litter_active = track && gen_mode() && litter_rules() && !compress_final_phase();
    // Slow compression (gen mode, non-final): accept equal-size candidates and splice the
    // largest allowed one, so the circuit shrinks only when a window has no same-size peer.
    let slow_active = gen_mode() && slow_compress() && !compress_final_phase();

    for _ in 0..trials {
        let t_trial = Instant::now();

        let (sub, start, end) = if litter_active {
            let mut best: Option<(usize, (CircuitSeq, usize, usize))> = None;
            for _ in 0..litter_window_samples() {
                let (s, st, en) = random_subcircuit(&compressed);
                if s.gates.is_empty() {
                    continue;
                }
                LITTER_WIN_CONSIDERED.fetch_add(1, Ordering::Relaxed);
                let (distinct, full) = window_litter_stats(&tags[st..en]);
                if full {
                    LITTER_BAN_FULL.fetch_add(1, Ordering::Relaxed);
                    continue;
                }
                if best.as_ref().map_or(true, |(d, _)| distinct > *d) {
                    best = Some((distinct, (s, st, en)));
                }
            }
            match best {
                Some((distinct, w)) => {
                    if distinct >= 2 {
                        LITTER_TIER1.fetch_add(1, Ordering::Relaxed);
                        LITTER_DISTINCT_SUM.fetch_add(distinct, Ordering::Relaxed);
                    } else {
                        LITTER_TIER2.fetch_add(1, Ordering::Relaxed);
                    }
                    w
                }
                None => continue, // every sampled window was banned or empty
            }
        } else {
            random_subcircuit(&compressed)
        };

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

        let fwd_key = xxh3_128(&polys_repr_blob(&fwd_polys))
            .to_le_bytes()
            .to_vec();
        let fwd_shard = fwd_key[0] as usize;

        let t_txn = Instant::now();
        let txn = match env.begin_ro_txn() {
            Ok(t) => t,
            Err(_) => continue,
        };
        TXN_TIME.fetch_add(t_txn.elapsed().as_nanos() as u64, Ordering::Relaxed);

        let t_lookup = Instant::now();
        let fwd_result = txn
            .get(shard_dbs[fwd_shard], &fwd_key)
            .map(|v: &[u8]| v.to_vec());
        LMDB_LOOKUP_TIME.fetch_add(t_lookup.elapsed().as_nanos() as u64, Ordering::Relaxed);

        let (value, final_order, is_reversed) = if let Ok(v) = fwd_result {
            (v, fwd_order, false)
        } else {
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
                    compressed.gates.len(),
                    start,
                    end,
                    sub.gates.len(),
                    sub.used_wires().len(),
                    canon2_elapsed as u128 / 1_000_000
                );
            }

            if rev_polys.is_empty() {
                continue;
            }

            let rev_key = xxh3_128(&polys_repr_blob(&rev_polys))
                .to_le_bytes()
                .to_vec();
            let rev_shard = rev_key[0] as usize;

            let t_lookup2 = Instant::now();
            let rev_result = txn
                .get(shard_dbs[rev_shard], &rev_key)
                .map(|v: &[u8]| v.to_vec());
            LMDB_LOOKUP_TIME.fetch_add(t_lookup2.elapsed().as_nanos() as u64, Ordering::Relaxed);

            match rev_result {
                Ok(v) => (v, rev_order, true),
                Err(_) => continue,
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
            // Slow mode also admits equal-size ("lateral") candidates; never larger.
            if candidate.gates.len() < sub.gates.len()
                || (slow_active && candidate.gates.len() == sub.gates.len())
            {
                candidates.push(candidate);
            }
        }
        FROM_BLOB_TIME.fetch_add(t_blob.elapsed().as_nanos() as u64, Ordering::Relaxed);

        if candidates.is_empty() {
            continue;
        }

        let min_gates = candidates.iter().map(|c| c.gates.len()).min().unwrap();
        let sub_len = sub.gates.len();
        // Slow mode inverts the size preference: splice the LARGEST allowed candidate (lateral
        // when one exists), so compression proceeds only where the DB forces it.
        let pick_gates = if slow_active {
            candidates.iter().map(|c| c.gates.len()).max().unwrap()
        } else {
            min_gates
        };
        let t_rewire = Instant::now();
        let repl: CircuitSeq = if gen_mode() {
            // gen mode (#9): among the smallest (most-compressing) equivalents — or the largest
            // allowed under SLOW_COMPRESS — rewire each into circuit-space, then let the incoming
            // ranker pick by GLOBAL fanout/leeway features (fanout-histogram-to-target,
            // low-leeway reduction, MAX_FANOUT cap).
            let min_set: Vec<CircuitSeq> = candidates
                .into_iter()
                .filter(|c| c.gates.len() == pick_gates)
                .map(|cand| rewire_candidate(cand, is_reversed, &final_order, &used, n, &mut rng))
                .collect();
            let left = &compressed.gates[..start];
            let right = &compressed.gates[end..];
            let feats: Vec<crate::replace::ranking::CandFeatures> = min_set
                .iter()
                .map(|cc| cand_features(&cc.gates, left, right))
                .collect();
            let order = crate::replace::ranking::incoming().order(&feats);
            let pick = order.first().copied().unwrap_or(0);
            min_set.into_iter().nth(pick).unwrap()
        } else {
            let mut best: Vec<CircuitSeq> = if sat_scoring_enabled() {
            let max_len = min_gates
                .saturating_add(sat_score_slack())
                .min(sub.gates.len() - 1);
            let seed = sat_score_seed();
            let base_score = compression_selection_score(&score_subcircuit(
                &sub.gates,
                sub.max_wire() + 1,
                seed ^ 0xc0de_5678,
            ));
            let scored: Vec<(f64, CircuitSeq)> = candidates
                .into_iter()
                .filter(|c| c.gates.len() <= max_len)
                .enumerate()
                .filter_map(|(idx, candidate)| {
                    let score_n = candidate.max_wire() + 1;
                    let sat_score = score_subcircuit(&candidate.gates, score_n, seed ^ idx as u64);
                    if sat_cone_aware_enabled()
                        && sat_score.output_cone_fraction < sat_cone_min_fraction()
                    {
                        return None;
                    }
                    if sat_bcp_enabled() && sat_score.bcp_resistance < sat_bcp_min_resistance() {
                        return None;
                    }
                    let candidate_score = compression_selection_score(&sat_score);
                    if sat_compress_protect_enabled()
                        && candidate_score + sat_compress_preserve_delta() < base_score
                    {
                        return None;
                    }
                    Some((candidate_score, candidate))
                })
                .collect();
            if scored.is_empty() {
                continue;
            }
            let max_score = scored
                .iter()
                .map(|(score, _)| *score)
                .fold(f64::NEG_INFINITY, f64::max);
            scored
                .into_iter()
                .filter(|(score, _)| (*score - max_score).abs() <= 1e-9)
                .map(|(_, candidate)| candidate)
                .collect()
            } else {
                candidates
                    .into_iter()
                    .filter(|c| c.gates.len() == min_gates)
                    .collect()
            };
            let idx = rng.random_range(0..best.len());
            let raw = best.swap_remove(idx);
            rewire_candidate(raw, is_reversed, &final_order, &used, n, &mut rng)
        };
        *COMPRESSION_HISTOGRAM
            .entry((sub_len as u8, repl.gates.len() as u8))
            .or_insert(0) += 1;
        REWIRE_TIME.fetch_add(t_rewire.elapsed().as_nanos() as u64, Ordering::Relaxed);

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
        // An equal-size candidate (slow mode) can rewire to the very window it came from —
        // splicing it back would be a no-op that still re-stamps litters. Skip those.
        if repl_len == end - start && repl.gates[..] == compressed.gates[start..end] {
            LITTER_IDENTITY_SKIP.fetch_add(1, Ordering::Relaxed);
            continue;
        }
        SWEEP_MOVES.fetch_add(1, Ordering::Relaxed);
        if repl_len == end - start {
            LITTER_LATERAL.fetch_add(1, Ordering::Relaxed);
            compressed.gates[start..end].copy_from_slice(&repl.gates);
        } else {
            LITTER_SHRINK.fetch_add(1, Ordering::Relaxed);
            compressed.gates.splice(start..end, repl.gates);
        }
        if track {
            // The replaced window [start, end) becomes `repl_len` freshly-created gates,
            // tagged Tag::NEW (survivor mode) or floor(median(window))+1 (gen mode).
            let nt = new_gate_tag(&tags[start..end], repl_len);
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
    env: &lmdb::Environment,
    shard_dbs: &[lmdb::Database],
    mode: usize,
    base_offset: usize,
    tags: &mut Vec<Tag>,
) -> CircuitSeq {
    // `tags` (non-empty when --track-survivors): per-gate origin id, kept in lockstep with
    // `circuit.gates`. `base_offset` is this chunk's offset in the full circuit.
    let track = !tags.is_empty();
    let mut circuit = c.clone();
    let mut rng = rand::rng();

    let mut i = 0;
    while i < circuit.gates.len().saturating_sub(1) {
        if circuit.gates[i] == circuit.gates[i + 1] {
            circuit.gates.drain(i..=i + 1);
            if track {
                tags.drain(i..=i + 1);
            }
            i = i.saturating_sub(2);
        } else {
            i += 1;
        }
    }

    let chunk_budget = compress_chunk_budget_ms();
    let chunk_started = Instant::now();
    for _ in 0..trials {
        // Straggler guard: stop starting trials once this chunk exceeds its wall-clock
        // budget, so one hard chunk cannot serialize the whole parallel sweep.
        if let Some(ms) = chunk_budget {
            if chunk_started.elapsed().as_millis() as u64 >= ms {
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
        let n_wires = used_wires.len();
        let max = num_wires;
        let new_wires = rng.random_range(n_wires..=max);
        if new_wires > n_wires {
            let mut count = n_wires;
            while count < new_wires {
                let random = rng.random_range(0..num_wires);
                if used_wires.contains(&(random as u16)) {
                    continue;
                }
                used_wires.push(random as u16);
                count += 1;
            }
        }

        let sub_num_wires = used_wires.len();

        let t4 = Instant::now();
        let sub_gates = subcircuit.gates.len();
        // Tags of the contiguous block [start, end]; compress_lmdb mutates them in lockstep.
        let mut block_tags: Vec<Tag> = if track {
            tags[start..=end].to_vec()
        } else {
            Vec::new()
        };
        let subcircuit_temp = compress_lmdb(
            &subcircuit,
            10,
            sub_num_wires,
            env,
            shard_dbs,
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
                "[compress-trace] slow compress_lmdb mode={} outer_gates={} outer_wires={} outer_span={} out_gates={} elapsed_ms={}",
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
    let mut i = 0;
    while i < circuit.gates.len().saturating_sub(1) {
        if circuit.gates[i] == circuit.gates[i + 1] {
            circuit.gates.drain(i..=i + 1);
            if track {
                tags.drain(i..=i + 1);
            }
            i = i.saturating_sub(2);
        } else {
            i += 1;
        }
    }
    DEDUP_TIME.fetch_add(t7.elapsed().as_nanos() as u64, Ordering::Relaxed);

    circuit
}

pub fn expand_big_ancillas<'a>(
    c: &CircuitSeq,
    trials: usize,
    num_wires: usize,
    env: &lmdb::Environment,
    shard_dbs: &[lmdb::Database],
    mode: usize,
    pair_mode: &ExpandPairMode<'a>,
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
        let cc = contiguous_convex(&mut circuit, &mut subcircuit_gates, num_wires, &mut Vec::new());
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
                ExpandPairMode::Curated { curated_shard_dbs } => {
                    use crate::replace::pairs::expand_curated_lmdb;
                    expand_curated_lmdb(
                        &[circuit.gates[start], circuit.gates[end]],
                        num_wires,
                        env,
                        curated_shard_dbs,
                        shard_dbs,
                    )
                }
                ExpandPairMode::Db => None, // handled by expand_lmdb below
            };
            if let Some(repl) = repl_opt {
                if repl.len() > 2 {
                    circuit.gates.splice(start..=end, repl);
                }
                REPLACE_TIME.fetch_add(t6.elapsed().as_nanos() as u64, Ordering::Relaxed);
                continue;
            }
            // Db mode falls through to expand_lmdb
            REPLACE_TIME.fetch_add(t6.elapsed().as_nanos() as u64, Ordering::Relaxed);
        }

        // --- 3-5 gate path (and 2-gate Db mode): use shard DB via expand_lmdb ---
        // Pass num_wires (full circuit wire count) so extra wires are assigned correctly.
        let t4 = Instant::now();
        let subcircuit_temp = expand_lmdb(
            &subcircuit,
            10,
            num_wires,
            env,
            shard_dbs,
            &ExpandPairMode::Db,
        );
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
    let txn = TXN_TIME.load(Ordering::Relaxed);
    let lmdb_lookup = LMDB_LOOKUP_TIME.load(Ordering::Relaxed);
    let from_blob = FROM_BLOB_TIME.load(Ordering::Relaxed);
    let trial = TRIAL_TIME.load(Ordering::Relaxed);
    let rule_l_time = CANON4_RULE_L_TIME.load(Ordering::Relaxed);
    let rule_l_calls = CANON4_RULE_L_CALLS.load(Ordering::Relaxed);
    let rule_l_branches = CANON4_RULE_L_BRANCHES.load(Ordering::Relaxed);

    let samf_made = SAMF_COMPRESSIONS_MADE.load(Ordering::Relaxed);
    let samf_failed = SAMF_COMPRESSIONS_FAILED.load(Ordering::Relaxed);

    let ns = 60_000_000_000.0f64;
    let threshold_ns = 15.0 * ns; // 15 minutes in nanoseconds

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
    if txn as f64 >= threshold_ns {
        println!("LMDB transaction begin time: {:.2} min", txn as f64 / ns);
    }
    if lmdb_lookup as f64 >= threshold_ns {
        println!("LMDB lookup time: {:.2} min", lmdb_lookup as f64 / ns);
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
mod float_tests {
    use super::*;
    use crate::circuit::circuit::{CircuitSeq, Gate};
    use rand::rngs::StdRng;
    use rand::SeedableRng;

    // float_all_gates must preserve the circuit's function and keep each gate's tag attached
    // to that gate through every slide.
    #[test]
    fn float_preserves_function_and_tag_pairing() {
        for seed in 0..40u64 {
            let mut rng = StdRng::seed_from_u64(seed);
            let n = 12u16;
            let mut gates: Vec<[u16; 3]> = Vec::new();
            // Distinct tag per gate via a unique low bit pattern we can track: store index in Tag.0.
            for k in 0..60u64 {
                use rand::Rng;
                let a = rng.random_range(0..n);
                let b = rng.random_range(0..n);
                let c = rng.random_range(0..n);
                if a != b && a != c && b != c {
                    gates.push([a, b, c]);
                    let _ = k;
                }
            }
            let mut tags: Vec<Tag> = (0..gates.len() as u64).map(Tag).collect();
            let before = CircuitSeq { gates: gates.clone() };
            // Map: which gate does each tag sit on (by gate content is ambiguous; track by
            // pairing invariant — tag i must always sit on the gate originally at index i).
            let orig: Vec<[u16; 3]> = gates.clone();
            float_all_gates(&mut gates, &mut tags, &mut rng);
            let after = CircuitSeq { gates: gates.clone() };
            assert!(
                before.probably_equal(&after, n as usize, 400).is_ok(),
                "float changed function (seed {})",
                seed
            );
            // Tag/gate pairing: the gate now carrying tag t must equal orig[t].
            for (pos, t) in tags.iter().enumerate() {
                assert_eq!(gates[pos], orig[t.0 as usize], "tag/gate desync (seed {})", seed);
            }
            // No collision was crossed: relative order of any two colliding gates is preserved.
            for x in 0..orig.len() {
                for y in (x + 1)..orig.len() {
                    if Gate::collides_index(&orig[x], &orig[y]) {
                        let px = tags.iter().position(|t| t.0 == x as u64).unwrap();
                        let py = tags.iter().position(|t| t.0 == y as u64).unwrap();
                        assert!(px < py, "colliding pair reordered (seed {})", seed);
                    }
                }
            }
        }
    }
}

#[cfg(test)]
mod degree_filter_tests {
    use super::*;
    use crate::circuit::circuit::CircuitSeq;
    use rand::rngs::StdRng;
    use rand::{Rng, SeedableRng};

    // True max ANF degree of the window (both directions), computed exactly via to_polynomial.
    fn true_max_degree(sub: &CircuitSeq) -> usize {
        let deg = |c: &CircuitSeq| -> usize {
            let n = c.max_wire() as usize + 1;
            c.to_polynomial(n, 0, c.gates.len())
                .iter()
                .flat_map(|p| p.iter().map(|m| m.count_ones() as usize))
                .max()
                .unwrap_or(0)
        };
        let rev = CircuitSeq { gates: sub.gates.iter().rev().copied().collect() };
        deg(sub).max(deg(&rev))
    }

    #[test]
    fn degree_filter_is_sound_and_effective() {
        let mut rng = StdRng::seed_from_u64(12345);
        let d = 5usize; // small threshold so tests exercise both sides quickly
        let mut discarded_ok = 0;
        let mut low_seen = 0;
        let mut high_seen = 0;
        let mut high_caught = 0;
        for _ in 0..4000 {
            let nw = rng.random_range(6..14u16);
            let ng = rng.random_range(4..16usize);
            let mut gates: Vec<[u16; 3]> = Vec::new();
            while gates.len() < ng {
                let a = rng.random_range(0..nw);
                let b = rng.random_range(0..nw);
                let c = rng.random_range(0..nw);
                if a != b && a != c && b != c {
                    gates.push([a, b, c]);
                }
            }
            let sub = CircuitSeq { gates };
            let truth = true_max_degree(&sub);
            let discards = degree_filter_discards(&sub, d, 10, &mut rng);
            // SOUNDNESS: never discard a window whose min-direction degree could match (<= d).
            // degree_filter_discards requires BOTH directions > d, so if EITHER direction <= d it
            // must not discard. Reconstruct per-direction to assert precisely.
            let fwd_deg = {
                let n = sub.max_wire() as usize + 1;
                sub.to_polynomial(n, 0, sub.gates.len()).iter()
                    .flat_map(|p| p.iter().map(|m| m.count_ones() as usize)).max().unwrap_or(0)
            };
            let rev_c = CircuitSeq { gates: sub.gates.iter().rev().copied().collect() };
            let rev_deg = {
                let n = rev_c.max_wire() as usize + 1;
                rev_c.to_polynomial(n, 0, rev_c.gates.len()).iter()
                    .flat_map(|p| p.iter().map(|m| m.count_ones() as usize)).max().unwrap_or(0)
            };
            if fwd_deg <= d || rev_deg <= d {
                assert!(!discards, "SOUNDNESS VIOLATION: discarded a matchable window (fwd_deg={}, rev_deg={}, d={})", fwd_deg, rev_deg, d);
                low_seen += 1;
            } else {
                high_seen += 1;
                if discards { high_caught += 1; discarded_ok += 1; }
            }
            let _ = truth;
        }
        println!("low(unmatchable-guard held)={} high={} high_caught={} ({:.0}%)",
                 low_seen, high_seen, high_caught, 100.0 * high_caught as f64 / high_seen.max(1) as f64);
        assert!(discarded_ok > 0, "filter never fired — probes ineffective");
        let _ = high_seen;
    }

    // A balanced product tree: pairs of disjoint fresh inputs multiplied up `levels` deep, so
    // the top monomial degree is 2^levels — the shape mixing manufactures and the pathology the
    // filter exists for. Gate [a,b,c] gives a' = a ^ 1 ^ b ^ bc, top term b*c; feeding disjoint
    // subtrees into b and c squares the degree each level.
    fn product_tree(levels: u32) -> (CircuitSeq, usize) {
        let leaves = 1usize << levels;
        let mut next = leaves as u16; // fresh target wires start above the input leaves
        let mut gates: Vec<[u16; 3]> = Vec::new();
        // level 1: pair leaves 2i,2i+1 into fresh targets
        let mut layer: Vec<u16> = Vec::new();
        let mut i = 0u16;
        while (i as usize) < leaves {
            let t = next; next += 1;
            gates.push([t, i, i + 1]);
            layer.push(t);
            i += 2;
        }
        // higher levels: pair disjoint sub-results
        while layer.len() > 1 {
            let mut nl = Vec::new();
            let mut j = 0;
            while j + 1 < layer.len() {
                let t = next; next += 1;
                gates.push([t, layer[j], layer[j + 1]]);
                nl.push(t);
                j += 2;
            }
            layer = nl;
        }
        (CircuitSeq { gates }, 1usize << levels)
    }

    // Effectiveness on the windows that actually matter: constructed product trees whose degree
    // (2^levels) is well above the threshold. These must be caught essentially always.
    #[test]
    fn degree_filter_catches_explosive_windows() {
        let mut rng = StdRng::seed_from_u64(999);
        let d = 5usize;
        let deg = |c: &CircuitSeq| -> usize {
            let n = c.max_wire() as usize + 1;
            c.to_polynomial(n, 0, c.gates.len()).iter()
                .flat_map(|p| p.iter().map(|m| m.count_ones() as usize)).max().unwrap_or(0)
        };
        // The windows that actually STALL canonicalization are the DENSE high-degree ones
        // (monomial count is what canon4 cost scales with; sparse high-degree windows are cheap).
        // Deeper product trees are both higher-degree and denser (level 4 ~ 33k monomials), so
        // the per-probe hit rate climbs steeply — that is exactly the regime the filter targets.
        // A product tree's inverse is low-degree (uncompute), so degree_filter_discards (both
        // directions) must NOT fire: reverse-compressible windows are preserved.
        let mono = |c: &CircuitSeq| -> usize {
            let n = c.max_wire() as usize + 1;
            c.to_polynomial(n, 0, c.gates.len()).iter().map(|p| p.len()).max().unwrap_or(0)
        };
        // Cap at level 4 (degree 16, ~33k monomials): level 5 would be degree 32, whose ground-
        // truth to_polynomial here explodes to billions of monomials — the very pathology the
        // filter sidesteps but the exact-degree oracle cannot. Level 4 already exercises the
        // dense-high regime the filter targets.
        for levels in [3u32, 4] {
            let (sub, top_deg) = product_tree(levels);
            assert!(deg(&sub) > d, "forward tree not high-degree");
            let trials = 200;
            let mut fwd_caught = 0;
            let mut wrongly_discarded = 0;
            for _ in 0..trials {
                if degree_exceeds_dir(&sub, false, d, 8, &mut rng) { fwd_caught += 1; }
                if degree_filter_discards(&sub, d, 8, &mut rng) { wrongly_discarded += 1; }
            }
            println!("tree deg {} ({} monomials): forward caught {}/{}, wrongly discarded {}",
                     top_deg, mono(&sub), fwd_caught, trials, wrongly_discarded);
            let rev = CircuitSeq { gates: sub.gates.iter().rev().copied().collect() };
            if deg(&rev) <= d {
                assert_eq!(wrongly_discarded, 0, "discarded a reverse-compressible window (deg {})", top_deg);
            }
            // The dense deep trees (the stall-causing regime) must be caught essentially always.
            if levels >= 4 {
                assert!(fwd_caught * 100 >= trials * 90, "dense deg-{} tree caught < 90%: {}/{}", top_deg, fwd_caught, trials);
            }
        }
    }
}
