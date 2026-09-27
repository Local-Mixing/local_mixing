//! extract_identities — reconstruct locally-geodesic (minimal) g57 identities
//! up to a target length from a frozen curated store.
//!
//! The curated store maps a function key to a chain of equivalent circuits.
//! Any two circuits `Ci`, `Cj` under one key compute the same function, so
//! `Ci · reverse(Cj)` is an identity (g57 gates are involutions, so a circuit's
//! inverse is its reversed gate list). Pairing every same-key circuit pair,
//! keeping the locally-geodesic ones up to `--max-len`, and canonicalising by
//! the dihedral+relabel orbit, reconstructs the store's minimal-identity set.
//!
//! Exhaustiveness note: the deployed store is a bounded rebuild (<= ~20 friends
//! per key). This extraction is therefore exhaustive UP TO that cap — a
//! function with more than ~20 minimal circuits could contribute a few
//! identities that the store never kept. Reported alongside the counts.
//!
//! Usage:
//!   extract_identities <store_dir> [--max-len 12] [--min-len 4]
//!                      [--native|--swap|--auto] [--out FILE] [--shards N]
//!
//! `--auto` (default) samples shard 0 and picks the control convention (native
//! vs swapped) under which the store's same-key pairs actually verify as
//! identities.

use local_mixing::circuit::{CircuitSeq, Gate as G57Gate};
use local_mixing::db_generation::curated_full::{dihedral_canonical_word, word_bytes};
use local_mixing::db_mixing::frozen::scan_shard;
use rayon::prelude::*;
use std::collections::BTreeMap;
use std::collections::HashMap;
use std::collections::HashSet;
use std::io::Write;

type Gate = [u16; 3];

/// Canonical form of a circuit: compact used wires to 0..k-1, then relabel by
/// the polynomial-canonical wire order. Two circuits computing the same
/// function (any wire labeling) land on the SAME canonical k-wire frame, so
/// they can be paired into a valid identity. Returns (function key, canonical
/// gates). None when canonicalization is skipped.
fn canonical_form(gates: &[Gate]) -> Option<([u8; 16], Vec<Gate>)> {
    let mut used: Vec<u16> = gates.iter().flat_map(|g| g.iter().copied()).collect();
    used.sort_unstable();
    used.dedup();
    let mut pos = HashMap::with_capacity(used.len());
    for (i, &w) in used.iter().enumerate() {
        pos.insert(w, i as u16);
    }
    let compact: Vec<Gate> = gates
        .iter()
        .map(|g| [pos[&g[0]], pos[&g[1]], pos[&g[2]]])
        .collect();
    let mut cs = CircuitSeq { gates: compact };
    let (key, order, _used) = cs.canonicalize_polys_single_hashed(false);
    let key = key?;
    if order.data.len() == used.len() && !order.data.is_empty() {
        cs.rewire(&order, used.len());
    }
    Some((key, cs.gates))
}

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

/// Width-independent probe identity test: a fixed xorshift state set masked to
/// the word's support. A non-identity permutation fixing all PROBES states is
/// astronomically unlikely (~2^-PROBES per independent state), so this never
/// falsely accepts in practice; used in the hot scan. Survivors are re-checked
/// exactly by `is_identity_exact` before output.
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

/// Exact identity test (all 2^w states) for small support; falls back to a wide
/// probe otherwise. Used only on the small deduplicated survivor set.
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
    // gates commute unless one writes a wire the other reads or writes
    g[0] != h[0] && g[0] != h[1] && g[0] != h[2] && h[0] != g[1] && h[0] != g[2]
}

/// No adjacent-equal gate and no equal pair separated only by commuting gates
/// (both are shuffle-cancellable, so the word is not reduced).
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

/// Locally geodesic (SMID): reduced, and no proper contiguous subword is itself
/// an identity.
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

fn identity_from(a: &[Gate], b: &[Gate]) -> Vec<Gate> {
    let mut v = Vec::with_capacity(a.len() + b.len());
    v.extend_from_slice(a);
    v.extend(b.iter().rev().copied());
    v
}

struct Args {
    dir: String,
    max_len: usize,
    min_len: usize,
    conv: Option<bool>, // Some(swap) or None=auto
    out: Option<String>,
    shards: Option<usize>,
    complete: bool, // reverse-completion: bucket by function key incl. reverses
}

fn parse_args() -> Args {
    let mut a = std::env::args().skip(1);
    let dir = a.next().expect("usage: extract_identities <store_dir> [opts]");
    let mut args = Args {
        dir,
        max_len: 12,
        min_len: 4,
        conv: None,
        out: None,
        shards: None,
        complete: false,
    };
    while let Some(tok) = a.next() {
        match tok.as_str() {
            "--max-len" => args.max_len = a.next().unwrap().parse().unwrap(),
            "--min-len" => args.min_len = a.next().unwrap().parse().unwrap(),
            "--native" => args.conv = Some(false),
            "--swap" => args.conv = Some(true),
            "--auto" => args.conv = None,
            "--out" => args.out = Some(a.next().unwrap()),
            "--shards" => args.shards = Some(a.next().unwrap().parse().unwrap()),
            "--complete" => args.complete = true,
            "--probe" => {}
            other => panic!("unknown arg {other}"),
        }
    }
    args
}

/// Reverse-completion pass: reconstruct the COMPLETE circuit set per function by
/// bucketing every stored (<=7-gate) circuit AND its reverse under their
/// canonical function keys (on a common wire frame), then tiered-pairing within
/// each completed bucket. Recovers the identities whose two halves the store
/// kept under different keys (C vs C*). Returns the canonical identity set.
fn run_complete(dir: &str, shards: &[usize], swap: bool, max_len: usize, min_len: usize)
    -> HashSet<Vec<u8>>
{
    // Per shard: canonical function key -> canonical circuits (len<=7).
    let per_shard: Vec<HashMap<[u8; 16], Vec<Vec<Gate>>>> = shards
        .par_iter()
        .map(|&s| {
            let mut map: HashMap<[u8; 16], Vec<Vec<Gate>>> = HashMap::new();
            scan_shard(dir, s, &mut |value: &[u8]| {
                for c in parse_value(value, swap) {
                    if c.len() > 7 || c.len() < 1 {
                        continue;
                    }
                    let rev: Vec<Gate> = c.iter().rev().copied().collect();
                    for w in [&c, &rev] {
                        if let Some((key, canon)) = canonical_form(w) {
                            map.entry(key).or_default().push(canon);
                        }
                    }
                }
            });
            eprintln!("[shard {s:02x}] {} function buckets", map.len());
            map
        })
        .collect();

    // Merge shard maps: a function f and its inverse f^-1 live in different
    // store shards, so buckets must be unioned globally.
    let mut global: HashMap<[u8; 16], Vec<Vec<Gate>>> = HashMap::new();
    for m in per_shard {
        for (k, v) in m {
            global.entry(k).or_default().extend(v);
        }
    }
    eprintln!("[complete] {} distinct function buckets merged", global.len());

    // Pair within each completed bucket (tiered kmin / kmin+1).
    let buckets: Vec<Vec<Vec<Gate>>> = global.into_values().collect();
    buckets
        .par_iter()
        .map(|circuits| {
            // dedup circuits in this bucket
            let mut seen = HashSet::new();
            let mut uniq: Vec<&Vec<Gate>> = Vec::new();
            for c in circuits {
                if seen.insert(word_bytes(c)) {
                    uniq.push(c);
                }
            }
            let mut set: HashSet<Vec<u8>> = HashSet::new();
            if uniq.len() < 2 {
                return set;
            }
            let kmin = uniq.iter().map(|c| c.len()).min().unwrap();
            if 2 * kmin > max_len {
                return set;
            }
            let mins: Vec<&Vec<Gate>> =
                uniq.iter().filter(|c| c.len() == kmin).copied().collect();
            let plus: Vec<&Vec<Gate>> =
                uniq.iter().filter(|c| c.len() == kmin + 1).copied().collect();
            let mut consider = |ident: Vec<Gate>, set: &mut HashSet<Vec<u8>>| {
                if ident.len() >= min_len
                    && ident.len() <= max_len
                    && is_identity(&ident)
                    && locally_geodesic(&ident)
                {
                    set.insert(word_bytes(&dihedral_canonical_word(&ident)));
                }
            };
            for i in 0..mins.len() {
                for j in (i + 1)..mins.len() {
                    consider(identity_from(mins[i], mins[j]), &mut set);
                }
            }
            if 2 * kmin + 1 <= max_len {
                for a in &mins {
                    for b in &plus {
                        consider(identity_from(a, b), &mut set);
                    }
                }
            }
            set
        })
        .reduce(HashSet::new, |mut a, b| {
            a.extend(b);
            a
        })
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

/// Probe the store's key/frame conventions: for a sample of values, does my
/// canonical key round-trip through the reader's get(), and are stored circuits
/// already on the canonical wire frame (order == identity)? Determines whether
/// the per-value+get completion is viable.
fn run_probe(dir: &str, swap: bool) {
    use local_mixing::db_mixing::frozen::FrozenDb;
    let db = FrozenDb::open(dir, None);
    let mut checked = 0u64;
    let mut roundtrip = 0u64;
    let mut order_identity = 0u64;
    let mut inv_hit = 0u64;
    let mut involutions = 0u64;
    for sh in 0..6usize {
    scan_shard(dir, sh, &mut |value: &[u8]| {
        if checked >= 12000 {
            return;
        }
        let cs = parse_value(value, swap);
        let Some(a1) = cs.first() else { return };
        checked += 1;
        // canonical key + order for the first circuit
        let mut used: Vec<u16> = a1.iter().flat_map(|g| g.iter().copied()).collect();
        used.sort_unstable();
        used.dedup();
        let mut pos = HashMap::new();
        for (i, &w) in used.iter().enumerate() {
            pos.insert(w, i as u16);
        }
        let compact: Vec<Gate> = a1.iter().map(|g| [pos[&g[0]], pos[&g[1]], pos[&g[2]]]).collect();
        let csq = CircuitSeq { gates: compact };
        let (key, order, _u) = csq.canonicalize_polys_single_hashed(false);
        let Some(key) = key else { return };
        if order.data.iter().enumerate().all(|(i, &v)| i == v) {
            order_identity += 1;
        }
        // round-trip: does get(key) return a value that decodes to circuits
        // including a1 (possibly relabeled)?
        if let Some(v2) = db.get_regular(&key) {
            let cs2 = parse_value(&v2, swap);
            // compare as canonical (compact) forms
            let a1c: Vec<Gate> = a1
                .iter()
                .map(|g| [pos[&g[0]], pos[&g[1]], pos[&g[2]]])
                .collect();
            let hit = cs2.iter().any(|c| {
                let mut u: Vec<u16> = c.iter().flat_map(|g| g.iter().copied()).collect();
                u.sort_unstable();
                u.dedup();
                let mut p = HashMap::new();
                for (i, &w) in u.iter().enumerate() {
                    p.insert(w, i as u16);
                }
                let cc: Vec<Gate> = c.iter().map(|g| [p[&g[0]], p[&g[1]], p[&g[2]]]).collect();
                cc == a1c
            });
            if hit {
                roundtrip += 1;
            }
        }
        // inverse key present? (and is f an involution => same key)
        let rev: Vec<Gate> = a1.iter().rev().copied().collect();
        if let Some((kinv, _)) = canonical_form(&rev) {
            if kinv == key {
                involutions += 1;
            } else if db.get_regular(&kinv).is_some() {
                inv_hit += 1;
            }
        }
    });
    }
    eprintln!(
        "[probe] {checked} values: key round-trips {roundtrip}, \
         order==identity {order_identity}, involutions (f==f^-1) {involutions}, \
         inverse-key f^-1 present as a SEPARATE store key {inv_hit}"
    );
}

fn main() {
    let args = parse_args();
    let swap = args.conv.unwrap_or_else(|| detect_convention(&args.dir, args.max_len));
    if std::env::args().any(|a| a == "--probe") {
        run_probe(&args.dir, swap);
        return;
    }
    eprintln!(
        "[extract] store={} max_len={} min_len={} convention={}",
        args.dir,
        args.max_len,
        args.min_len,
        if swap { "swapped-controls" } else { "native" }
    );
    let shards = shard_list(&args.dir, args.shards);
    eprintln!("[extract] scanning {} shards (mode: {})",
        shards.len(), if args.complete { "reverse-completed" } else { "within-key" });

    let mut all: HashSet<Vec<u8>> = if args.complete {
        run_complete(&args.dir, &shards, swap, args.max_len, args.min_len)
    } else {
        within_key(&args, swap, &shards)
    };

    // Exact re-verification of the (probe-accepted) survivor set.
    let before = all.len();
    all.retain(|wb| is_identity_exact(&bytes_to_gates(wb)));
    if all.len() != before {
        eprintln!(
            "[verify] {} of {before} survivors failed exact re-check (dropped)",
            before - all.len()
        );
    }
    let mut by_len: BTreeMap<usize, Vec<&Vec<u8>>> = BTreeMap::new();
    for w in &all {
        by_len.entry(w.len() / 6).or_default().push(w);
    }
    eprintln!("\n== distinct locally-geodesic identities by length ==");
    for (len, v) in &by_len {
        eprintln!("  {len:>3} gates : {}", v.len());
    }
    eprintln!("  TOTAL     : {}", all.len());
    if let Some(path) = args.out {
        let mut f = std::io::BufWriter::new(std::fs::File::create(&path).unwrap());
        writeln!(f, "length,n_wires,circuit").unwrap();
        for (len, v) in &by_len {
            for wb in v {
                let gates = bytes_to_gates(wb);
                let nw = width(&gates);
                let s: Vec<String> = gates
                    .iter()
                    .map(|g| format!("{},{},{}", g[0], g[1], g[2]))
                    .collect();
                writeln!(f, "{len},{nw},{}", s.join(";")).unwrap();
            }
        }
        eprintln!("[extract] wrote {path}");
    }
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

fn within_key(args: &Args, swap: bool, shards: &[usize]) -> HashSet<Vec<u8>> {
    // Tiered pairing (per store entry, kmin = its minimal circuit length):
    //   * 2*kmin-gate identities  = pairs of kmin-length (minimal) circuits;
    //   * (2*kmin+1)-gate identities = a kmin-length circuit paired with a
    //     (kmin+1)-length circuit (a permutation with both a (k-1)- and a
    //     k-gate circuit, k = kmin+1).
    // Pairing two non-minimal circuits is never locally geodesic (the shorter
    // circuit shortens the identity), so those pairs are skipped entirely.
    let per_shard: Vec<HashSet<Vec<u8>>> = shards
        .par_iter()
        .map(|&s| {
            let mut set: HashSet<Vec<u8>> = HashSet::new();
            scan_shard(&args.dir, s, &mut |value: &[u8]| {
                let cs = parse_value(value, swap);
                if cs.len() < 2 {
                    return;
                }
                let kmin = cs.iter().map(|c| c.len()).min().unwrap();
                if kmin < 1 || 2 * kmin > args.max_len {
                    return; // even tier already exceeds the cap
                }
                let mins: Vec<&Vec<Gate>> = cs.iter().filter(|c| c.len() == kmin).collect();
                let plus: Vec<&Vec<Gate>> =
                    cs.iter().filter(|c| c.len() == kmin + 1).collect();
                let consider = |ident: Vec<Gate>, set: &mut HashSet<Vec<u8>>| {
                    if ident.len() >= args.min_len
                        && ident.len() <= args.max_len
                        && is_identity(&ident)
                        && locally_geodesic(&ident)
                    {
                        set.insert(word_bytes(&dihedral_canonical_word(&ident)));
                    }
                };
                // 2*kmin-gate identities
                for i in 0..mins.len() {
                    for j in (i + 1)..mins.len() {
                        consider(identity_from(mins[i], mins[j]), &mut set);
                    }
                }
                // (2*kmin+1)-gate identities
                if 2 * kmin + 1 <= args.max_len {
                    for a in &mins {
                        for b in &plus {
                            consider(identity_from(a, b), &mut set);
                        }
                    }
                }
            });
            eprintln!("[shard {s:02x}] {} canonical identities so far", set.len());
            set
        })
        .collect();

    let mut all: HashSet<Vec<u8>> = HashSet::new();
    for s in per_shard {
        all.extend(s);
    }
    all
}
