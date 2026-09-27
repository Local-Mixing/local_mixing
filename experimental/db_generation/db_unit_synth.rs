//! db_unit_synth — DB-backed g57 "unit" synthesizer + effective-diameter probe.
//!
//! Reads a batch of g57 WORDS (one per line, gates "a,x,y a,x,y ..."; wires are
//! small, support <= ~6). For each word T it tries to realize the SAME
//! permutation as a short g57 word looked up in the frozen `regular` store
//! (the g57 ball of diameter 6 + partial 7..9), so the huge MMD units can be
//! replaced by near-geodesic DB words.
//!
//! Two attempts, in order (unified as "suffix radius" r):
//!   (a) DIRECT (r = 0): canonicalize T (forward AND reversed), get_regular; on
//!       hit, decode the value chain, relabel each member back to T's wire
//!       frame, record the shortest member's gate count.
//!   (b) ASYMMETRIC MITM (r = 1,2,3): BFS-enumerate suffix words s (deduped by
//!       the residual permutation on T's support, so the count is bounded), form
//!       the residual prefix circuit P = T . reverse(s) (append reverse(s) to T;
//!       g57 gates are involutions so this strips s), canonicalize+lookup P; on
//!       hit, unit = decoded_prefix ++ s, length = |prefix| + |s|. Keep the
//!       shortest over all s and record the smallest r that solved it.
//!
//! Every synthesized unit is re-verified (exact over 2^width states) to equal T.
//!
//! Usage:
//!   db_unit_synth <words.txt> [--rs 3] [--cap 300000] [--out per_unit.tsv]
//!                 [--chains chains.tsv] [--label NAME]
//! Env: FROZEN_DB_DIR (required), FROZEN_CURATED_DIR (optional, unused here).

use local_mixing::circuit::{CircuitSeq, Gate as G57Gate, Permutation};
use local_mixing::db_generation::curated_full::decode_legacy_value;
use local_mixing::db_mixing::frozen::FrozenDb;
use rand::rngs::StdRng;
use rand::SeedableRng;
use rayon::prelude::*;
use std::collections::{HashMap, HashSet};
use std::io::Write;
use std::sync::atomic::{AtomicU64, Ordering};
use std::time::Instant;

type Gate = [u16; 3];

// ---------------------------------------------------------------------------
// g57 permutation tables on a small compact wire frame (support <= 6 -> <= 64
// states). Matches src/circuit/xgate.rs / tools/pr_identity/g57.py: a gate
// (a,x,y) sets  s[a] ^= ( (s>>x)&1 ) | ( 1 - ((s>>y)&1) ) = x OR NOT y.
// ---------------------------------------------------------------------------
#[inline]
fn apply_gate_state(g: Gate, s: usize) -> usize {
    let (a, x, y) = (g[0] as usize, g[1] as usize, g[2] as usize);
    let fire = ((s >> x) & 1) | (1 - ((s >> y) & 1));
    s ^ (fire << a)
}

/// All g57 gates on m compact wires with three DISTINCT wires (the generator
/// draws a,x,y = rng.sample(block,3), so gates always have distinct wires).
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

/// Compact a word's wires to first-seen 0..m-1. Returns (compact_word, order)
/// where order[i] = original wire for compact wire i.
fn compact(word: &[Gate]) -> (Vec<Gate>, Vec<u16>) {
    let mut seen: HashMap<u16, u16> = HashMap::new();
    let mut order: Vec<u16> = Vec::new();
    let mut out = Vec::with_capacity(word.len());
    for g in word {
        let mut ng = [0u16; 3];
        for k in 0..3 {
            let w = g[k];
            let c = *seen.entry(w).or_insert_with(|| {
                order.push(w);
                (order.len() - 1) as u16
            });
            ng[k] = c;
        }
        out.push(ng);
    }
    (out, order)
}

fn max_wire(word: &[Gate]) -> usize {
    word.iter().flat_map(|g| g.iter().copied()).max().unwrap_or(0) as usize
}

/// Exact permutation table over 2^w states for a word on wires 0..w-1.
fn word_table(word: &[Gate], w: usize) -> Vec<u32> {
    let n = 1usize << w;
    let mut t: Vec<u32> = (0..n as u32).collect();
    // apply left-to-right: new[s] = g(old[s])
    for &g in word {
        for s in 0..n {
            t[s] = apply_gate_state(g, t[s] as usize) as u32;
        }
    }
    t
}

/// True iff words `a` and `b` compute the same permutation over 2^w states.
fn words_equal(a: &[Gate], b: &[Gate]) -> bool {
    let w = max_wire(a).max(max_wire(b)) + 1;
    if w > 24 {
        // fall back to probe (should not happen for units on <=6 wires)
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
// Relabel a decoded DB candidate (canonical wire space, forward orientation)
// back to the query's wire frame. Mirrors src/db_mixing/replace.rs
// rewire_candidate exactly.
// ---------------------------------------------------------------------------
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
        &Permutation { data: order_data.clone() },
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
        use rand::seq::SliceRandom;
        available.shuffle(rng);
        let mut avail = available.into_iter();
        while used_ext.len() < repl_n_b {
            match avail.next() {
                Some(w) => used_ext.push(w),
                // safety net: never let unrewire index past the map (frame_n
                // exhausted) — keep appending fresh high wires.
                None => used_ext.push((n + used_ext.len()) as u16),
            }
        }
    }
    CircuitSeq::unrewire_subcircuit(&repl, &used_ext).gates
}

/// Decode the value chain for a hit key into candidate words, relabeled back to
/// the query wire frame. Returns all chain members (each a full g57 word).
fn decode_and_relabel(
    value: &[u8],
    is_reversed: bool,
    order: &Permutation,
    used: &[u16],
    frame_n: usize,
    seed: u64,
) -> Vec<Vec<Gate>> {
    let mut out = Vec::new();
    let blobs = match decode_legacy_value(value) {
        Ok(b) => b,
        Err(_) => return out,
    };
    let mut rng = StdRng::seed_from_u64(seed);
    for blob in blobs {
        if blob.is_empty() || blob.len() % 3 != 0 {
            continue;
        }
        let cand = CircuitSeq::from_blob(&blob);
        let w = rewire_candidate(cand, is_reversed, order, used, frame_n, &mut rng);
        out.push(w);
    }
    out
}

// ---------------------------------------------------------------------------
// Lookup helper: canonicalize P (forward then reversed) and get_regular. On the
// first hit return (value, order, used, is_reversed).
// ---------------------------------------------------------------------------
fn lookup(
    db: &FrozenDb,
    p: &CircuitSeq,
) -> Option<(Vec<u8>, Permutation, Vec<u16>, bool)> {
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

// ---------------------------------------------------------------------------
// Per-unit result
// ---------------------------------------------------------------------------
struct UnitResult {
    idx: usize,
    input_len: usize,
    support: usize,
    best_len: i64,   // shortest synthesized unit length, -1 if unsolved
    best_r: i64,     // smallest suffix radius that solved it, -1 if unsolved
    best_width: i64, // wire count of the shortest synthesized unit (support+ancilla)
    direct_len: i64, // r=0 shortest length, -1 if direct miss
    solved_by: [bool; 4], // solved_by[r] = solvable using suffix radius <= r
    len_at_r: [i64; 4],   // shortest unit length achievable at radius exactly-<=r
    verify_fail: u64,
    capped: bool,
    best_word: Vec<Gate>, // the shortest synthesized unit word (for born-bare)
    // chain members for the P-key that produced best_word (all equal to T),
    // only populated when chains output requested:
    chain: Vec<Vec<Gate>>,
}

fn solve_unit(
    db: &FrozenDb,
    idx: usize,
    t_orig: &[Gate],
    rs_max: usize,
    cap: usize,
    frame_n: usize,
    want_chain: bool,
) -> UnitResult {
    let (tc, order_wires) = compact(t_orig);
    let m = order_wires.len();
    let mut res = UnitResult {
        idx,
        input_len: t_orig.len(),
        support: m,
        best_len: -1,
        best_r: -1,
        best_width: -1,
        direct_len: -1,
        solved_by: [false; 4],
        len_at_r: [-1; 4],
        verify_fail: 0,
        capped: false,
        best_word: Vec::new(),
        chain: Vec::new(),
    };
    if m < 2 {
        return res;
    }

    // Incremental BFS over suffix permutations on the compact frame, deduped by
    // the residual permutation table. Each perm is looked up as it is DEQUEUED
    // (FIFO => nondecreasing depth => the first hit is the smallest radius R_s),
    // and expansion is interleaved so a huge deep layer is never materialized up
    // front. Stops at the FIRST solving suffix; that hit's decoded chain gives a
    // near-geodesic length (<= geodesic(P) + R_s). rs<=2 layers are exhausted
    // well within any sane cap (|layer2| ~ m!(...) << cap), so rs<=2 coverage is
    // EXACT; only the (huge) radius-3 layer can be truncated by `cap`.
    let nstates = 1usize << m;
    let gates = gates_on(m);
    let id_perm: Vec<u8> = (0..nstates as u8).collect();
    let mut seen: HashSet<Vec<u8>> = HashSet::new();
    seen.insert(id_perm.clone());
    let mut queue: std::collections::VecDeque<(Vec<u8>, Vec<Gate>, usize)> =
        std::collections::VecDeque::new();
    queue.push_back((id_perm, Vec::new(), 0));

    let mut best_len = i64::MAX;
    let mut best_word: Vec<Gate> = Vec::new();
    let mut best_width = i64::MAX;
    let mut best_chain: Vec<Vec<Gate>> = Vec::new();

    while let Some((perm, s, depth)) = queue.pop_front() {
        // P = T . reverse(s)   (compact frame)
        let mut pgates = tc.clone();
        for &g in s.iter().rev() {
            pgates.push(g);
        }
        let p = CircuitSeq { gates: pgates };
        if let Some((value, ord, used, is_rev)) = lookup(db, &p) {
            let members = decode_and_relabel(
                &value,
                is_rev,
                &ord,
                &used,
                frame_n,
                (idx as u64) << 20 ^ (depth as u64) << 4,
            );
            let mut solved_here = false;
            for prefix in &members {
                // unit = prefix ++ s  (both on compact frame)
                let mut unit = prefix.clone();
                unit.extend_from_slice(&s);
                if !words_equal(&unit, &tc) {
                    res.verify_fail += 1;
                    continue;
                }
                solved_here = true;
                let ulen = unit.len() as i64;
                let uwidth = (max_wire(&unit) + 1) as i64;
                if ulen < best_len || (ulen == best_len && uwidth < best_width) {
                    best_len = ulen;
                    best_width = uwidth;
                    best_word = unit.clone();
                }
            }
            if solved_here {
                res.best_r = depth as i64;
                res.solved_by[depth] = true;
                res.len_at_r[depth] = best_len;
                if depth == 0 {
                    res.direct_len = best_len;
                }
                if want_chain {
                    for prefix in &members {
                        let mut unit = prefix.clone();
                        unit.extend_from_slice(&s);
                        if words_equal(&unit, &tc) {
                            best_chain.push(unit);
                        }
                    }
                }
                break; // first-hit stop: smallest R_s found
            }
        }
        // expand to depth+1 (interleaved; never build a deep layer up front)
        if depth < rs_max && seen.len() < cap {
            for &g in &gates {
                let mut np = vec![0u8; nstates];
                for st in 0..nstates {
                    np[st] = apply_gate_state(g, perm[st] as usize) as u8;
                }
                if seen.insert(np.clone()) {
                    let mut nw = s.clone();
                    nw.push(g);
                    queue.push_back((np, nw, depth + 1));
                    if seen.len() >= cap {
                        res.capped = true;
                        break;
                    }
                }
            }
        } else if seen.len() >= cap {
            res.capped = true;
        }
    }

    // make solved_by cumulative (solvable within radius <= r)
    let mut acc = false;
    for r in 0..4 {
        acc = acc || res.solved_by[r];
        res.solved_by[r] = acc;
        if acc && res.len_at_r[r] < 0 {
            // fill forward the best known length
            res.len_at_r[r] = best_len;
        }
    }

    if best_len != i64::MAX {
        res.best_len = best_len;
        res.best_width = best_width;
        // map best_word from compact frame back to ORIGINAL wires
        res.best_word = best_word.iter().map(|g| map_back(g, &order_wires, frame_n)).collect();
        if want_chain {
            res.chain = best_chain
                .iter()
                .map(|u| u.iter().map(|g| map_back(g, &order_wires, frame_n)).collect())
                .collect();
        }
    }
    res
}

/// Map a compact-frame gate back to original wires. Compact wires 0..m-1 map to
/// order_wires[i]; any ancilla wire >= m (introduced by a DB candidate) is
/// shifted above the original block into [frame_n, ...) so it can't collide
/// with the block wires.
fn map_back(g: &Gate, order_wires: &[u16], frame_n: usize) -> Gate {
    let m = order_wires.len();
    let mp = |w: u16| -> u16 {
        let wu = w as usize;
        if wu < m {
            order_wires[wu]
        } else {
            (frame_n + (wu - m)) as u16
        }
    };
    [mp(g[0]), mp(g[1]), mp(g[2])]
}

fn parse_word(line: &str) -> Vec<Gate> {
    let mut w = Vec::new();
    for tok in line.split_whitespace() {
        let parts: Vec<&str> = tok.split(',').collect();
        if parts.len() != 3 {
            continue;
        }
        let a: u16 = parts[0].parse().unwrap();
        let x: u16 = parts[1].parse().unwrap();
        let y: u16 = parts[2].parse().unwrap();
        w.push([a, x, y]);
    }
    w
}

fn fmt_word(w: &[Gate]) -> String {
    w.iter()
        .map(|g| format!("{},{},{}", g[0], g[1], g[2]))
        .collect::<Vec<_>>()
        .join(" ")
}

fn pct(a: usize, b: usize) -> f64 {
    if b == 0 {
        0.0
    } else {
        100.0 * a as f64 / b as f64
    }
}

fn main() {
    let mut args = std::env::args().skip(1);
    let input = args.next().expect(
        "usage: db_unit_synth <words.txt> [--rs 3] [--cap N] [--out F] [--chains F] [--label NAME]",
    );
    let mut rs_max = 3usize;
    let mut cap = 300_000usize;
    let mut out: Option<String> = None;
    let mut chains: Option<String> = None;
    let mut label = input.clone();
    let mut frame_n = 16usize; // wire frame for ancilla fill; block is 0..5
    while let Some(a) = args.next() {
        match a.as_str() {
            "--rs" => rs_max = args.next().unwrap().parse().unwrap(),
            "--cap" => cap = args.next().unwrap().parse().unwrap(),
            "--out" => out = Some(args.next().unwrap()),
            "--chains" => chains = Some(args.next().unwrap()),
            "--label" => label = args.next().unwrap(),
            "--frame-n" => frame_n = args.next().unwrap().parse().unwrap(),
            other => panic!("unknown arg {other}"),
        }
    }
    let rs_max = rs_max.min(3);

    let db = FrozenDb::from_env();

    let lines: Vec<String> = std::fs::read_to_string(&input)
        .expect("read input")
        .lines()
        .map(|s| s.trim().to_string())
        .filter(|s| !s.is_empty())
        .collect();
    let words: Vec<Vec<Gate>> = lines.iter().map(|l| parse_word(l)).collect();
    let n = words.len();
    eprintln!("[db_unit_synth] {label}: {n} words, rs_max={rs_max}, cap={cap}");

    let want_chain = chains.is_some();
    let t0 = Instant::now();
    let total_ns = AtomicU64::new(0);
    let mut results: Vec<UnitResult> = words
        .par_iter()
        .enumerate()
        .map(|(i, w)| {
            let ti = Instant::now();
            let r = solve_unit(&db, i, w, rs_max, cap, frame_n, want_chain);
            total_ns.fetch_add(ti.elapsed().as_nanos() as u64, Ordering::Relaxed);
            r
        })
        .collect();
    results.sort_by_key(|r| r.idx);
    let wall = t0.elapsed().as_secs_f64();

    // aggregate
    let solved_r: [usize; 4] = {
        let mut c = [0usize; 4];
        for r in 0..4 {
            c[r] = results.iter().filter(|u| u.solved_by[r]).count();
        }
        c
    };
    let direct = results.iter().filter(|u| u.direct_len >= 0).count();
    let any_solved = solved_r[rs_max];
    let verify_fail: u64 = results.iter().map(|u| u.verify_fail).sum();
    let capped = results.iter().filter(|u| u.capped).count();
    let cpu_ms = total_ns.load(Ordering::Relaxed) as f64 / 1e6;

    // best-r histogram
    let mut best_r_hist = [0usize; 5]; // 0,1,2,3, unsolved(4)
    for u in &results {
        if u.best_r < 0 {
            best_r_hist[4] += 1;
        } else {
            best_r_hist[u.best_r as usize] += 1;
        }
    }

    // length distribution over solved units (best_len), and <=15 count
    let mut lens: Vec<i64> = results.iter().filter(|u| u.best_len >= 0).map(|u| u.best_len).collect();
    lens.sort_unstable();
    let (lmin, lmed, lmax) = if lens.is_empty() {
        (-1, -1, -1)
    } else {
        (lens[0], lens[lens.len() / 2], lens[lens.len() - 1])
    };
    let le15 = results.iter().filter(|u| u.best_len >= 0 && u.best_len <= 15).count();
    let le12 = results.iter().filter(|u| u.best_len >= 0 && u.best_len <= 12).count();
    let widths_gt: usize = results.iter().filter(|u| u.best_width > u.support as i64).count();

    println!("==== {label} ====");
    println!("words                 : {n}");
    println!(
        "direct (r=0) solved   : {direct}  ({:.1}%)",
        pct(direct, n)
    );
    println!(
        "+MITM r<=1            : {}  ({:.1}%)",
        solved_r[1.min(rs_max)],
        pct(solved_r[1.min(rs_max)], n)
    );
    println!(
        "+MITM r<=2            : {}  ({:.1}%)",
        solved_r[2.min(rs_max)],
        pct(solved_r[2.min(rs_max)], n)
    );
    println!(
        "+MITM r<=3            : {}  ({:.1}%)",
        solved_r[3.min(rs_max)],
        pct(solved_r[3.min(rs_max)], n)
    );
    println!("total solved          : {any_solved}  ({:.1}%)", pct(any_solved, n));
    println!("best-r histogram      : r0={} r1={} r2={} r3={} UNSOLVED={}",
        best_r_hist[0], best_r_hist[1], best_r_hist[2], best_r_hist[3], best_r_hist[4]);
    println!("unit length (solved)  : min={lmin} median={lmed} max={lmax}");
    println!("length <=12 / <=15    : {le12} ({:.1}%) / {le15} ({:.1}%)",
        pct(le12, n), pct(le15, n));
    println!("solved units wider than support : {widths_gt}");
    println!("verify failures       : {verify_fail}   capped units: {capped}");
    println!("wall: {wall:.1}s   cpu: {cpu_ms:.0}ms   mean cpu/unit: {:.2}ms",
        cpu_ms / n as f64);

    if let Some(path) = out {
        let mut f = std::io::BufWriter::new(std::fs::File::create(&path).unwrap());
        writeln!(f, "idx\tinput_len\tsupport\tdirect_len\tbest_r\tbest_len\tbest_width\tsolved0\tsolved1\tsolved2\tsolved3\tword").unwrap();
        for u in &results {
            writeln!(
                f,
                "{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}",
                u.idx, u.input_len, u.support, u.direct_len, u.best_r, u.best_len, u.best_width,
                u.solved_by[0] as u8, u.solved_by[1] as u8, u.solved_by[2] as u8, u.solved_by[3] as u8,
                fmt_word(&u.best_word)
            ).unwrap();
        }
        eprintln!("[db_unit_synth] wrote {path}");
    }

    if let Some(path) = chains {
        let mut f = std::io::BufWriter::new(std::fs::File::create(&path).unwrap());
        writeln!(f, "idx\tmember_len\tword").unwrap();
        for u in &results {
            for member in &u.chain {
                writeln!(f, "{}\t{}\t{}", u.idx, member.len(), fmt_word(member)).unwrap();
            }
        }
        eprintln!("[db_unit_synth] wrote {path}");
    }
}
