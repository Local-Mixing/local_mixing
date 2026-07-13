//! Validation tests for the DB-generation pipeline, converted from the
//! standalone bins validate_dedup / validate_2rocks / verify_merge_prefix.
//!
//! All three need real rocks DBs on disk, so they are #[ignore]d by default.
//! Run them explicitly from a directory containing the DBs (or with the env
//! vars set), e.g. on the dbgen server tree:
//!
//!   cargo test --release -p local_mixing -- --ignored --nocapture \
//!       fresh_wire_dedup_kv_sets_match          # needs ./rocks_db_m7
//!       capped_2rocks_enumeration_matches_full  # needs ./rocks_db_m2 + m3
//!
//!   MERGE_PAR_DB=<out> MERGE_SOURCES=<src1>:<src2> MERGE_TEST_PREFIX=b0b1 \
//!   cargo test --release -- --ignored --nocapture merge_prefix_rebuild_matches

use std::collections::{BTreeMap, BTreeSet, HashMap};

use super::{
    abstract_gates_for_circuit_filtered, apply_wire_mapping, count_mappings_pruned,
    for_each_mapping, for_each_mapping_capped, open_db_for_read,
};
use crate::circuit::circuit::{CircuitSeq, canonicalize_polys_4, polys_repr_blob};
use rayon::prelude::*;
use xxhash_rust::xxh3::xxh3_128;

// ------------------------------------------------------------ shared helpers

fn ordered2(n: usize) -> usize {
    if n >= 2 { n * (n - 1) } else { 0 }
}

fn ordered3(n: usize) -> usize {
    if n >= 3 { n * (n - 1) * (n - 2) } else { 0 }
}

fn touched_wires(circuit: &CircuitSeq) -> Vec<u16> {
    let mut touched: Vec<u16> = Vec::new();
    for gate in &circuit.gates {
        for &w in gate.iter() {
            if !touched.contains(&w) {
                touched.push(w);
            }
        }
    }
    touched.sort();
    touched
}

fn encode_circuit(circuit_blob: &[u8]) -> Vec<u8> {
    let mut v = Vec::with_capacity(1 + circuit_blob.len());
    v.push(circuit_blob.len() as u8);
    v.extend_from_slice(circuit_blob);
    v
}

// ============================================================ dedup validation
// Fresh-wire dedup patch to abstract_gates_for_circuit_filtered: for real
// rows from rocks_db_m7, the ORIGINAL (pre-dedup) filtered expansion and the
// NEW dedup expansion must produce identical (key, value) sets through the
// exact build_from_rocks worker pipeline, and emitted + skipped must account
// for every ordered gate triple.

const DEDUP_M: usize = 8;
const DEDUP_N: usize = 3 * DEDUP_M; // 24 wires
const DEDUP_MIN_N: usize = 17;

fn expand_abstract_gate(gate: [u16; 3], untouched: &[u16]) -> Vec<[u16; 3]> {
    const UNUSED: u16 = 512;
    let slots: Vec<usize> = gate
        .iter()
        .enumerate()
        .filter(|(_, w)| **w == UNUSED)
        .map(|(i, _)| i)
        .collect();
    let mut result = Vec::new();
    match slots.len() {
        0 => result.push(gate),
        1 => {
            for &u in untouched {
                let mut g = gate;
                g[slots[0]] = u;
                result.push(g);
            }
        }
        2 => {
            for &u1 in untouched {
                for &u2 in untouched {
                    if u1 == u2 {
                        continue;
                    }
                    let mut g = gate;
                    g[slots[0]] = u1;
                    g[slots[1]] = u2;
                    result.push(g);
                }
            }
        }
        3 => {
            for &u1 in untouched {
                for &u2 in untouched {
                    if u2 == u1 {
                        continue;
                    }
                    for &u3 in untouched {
                        if u3 == u1 || u3 == u2 {
                            continue;
                        }
                        result.push([u1, u2, u3]);
                    }
                }
            }
        }
        _ => unreachable!(),
    }
    result
}

/// Exact copy of abstract_gates_for_circuit_filtered as it was before the
/// fresh-wire dedup patch (full concrete expansion of every class).
fn old_filtered(
    circuit: &CircuitSeq,
    n: usize,
    min_n: usize,
    max_n: usize,
) -> (Vec<[u16; 3]>, usize) {
    const UNUSED: u16 = 512;

    let touched = touched_wires(circuit);
    let untouched: Vec<u16> = (0..n as u16).filter(|w| !touched.contains(w)).collect();

    let old_used = touched.len();
    let mut result = Vec::new();
    let mut skipped = 0usize;

    let allowed = |new_wires: usize| {
        let used = old_used + new_wires;
        used >= min_n && (max_n == 0 || used <= max_n)
    };

    if allowed(0) {
        for &a in &touched {
            for &b in &touched {
                if b == a {
                    continue;
                }
                for &c in &touched {
                    if c == a || c == b {
                        continue;
                    }
                    result.push([a, b, c]);
                }
            }
        }
    } else {
        skipped += ordered3(touched.len());
    }

    if !untouched.is_empty() {
        let count = 3 * ordered2(touched.len()) * untouched.len();
        if allowed(1) {
            for &b in &touched {
                for &c in &touched {
                    if c == b {
                        continue;
                    }
                    result.extend(expand_abstract_gate([UNUSED, b, c], &untouched));
                }
            }
            for &a in &touched {
                for &c in &touched {
                    if c == a {
                        continue;
                    }
                    result.extend(expand_abstract_gate([a, UNUSED, c], &untouched));
                }
            }
            for &a in &touched {
                for &b in &touched {
                    if b == a {
                        continue;
                    }
                    result.extend(expand_abstract_gate([a, b, UNUSED], &untouched));
                }
            }
        } else {
            skipped += count;
        }
    }

    if untouched.len() >= 2 {
        let count = 3 * touched.len() * ordered2(untouched.len());
        if allowed(2) {
            for &a in &touched {
                result.extend(expand_abstract_gate([a, UNUSED, UNUSED], &untouched));
            }
            for &b in &touched {
                result.extend(expand_abstract_gate([UNUSED, b, UNUSED], &untouched));
            }
            for &c in &touched {
                result.extend(expand_abstract_gate([UNUSED, UNUSED, c], &untouched));
            }
        } else {
            skipped += count;
        }
    }

    if untouched.len() >= 3 {
        let count = ordered3(untouched.len());
        if allowed(3) {
            result.extend(expand_abstract_gate([UNUSED, UNUSED, UNUSED], &untouched));
        } else {
            skipped += count;
        }
    }

    (result, skipped)
}

/// Replicates the per-candidate worker pipeline from build_from_rocks and
/// returns the set of (key, value) pairs that would be sent to the writer.
fn candidate_set(circuit: &CircuitSeq, gates: &[[u16; 3]]) -> BTreeSet<(Vec<u8>, Vec<u8>)> {
    let mut set = BTreeSet::new();
    for g in gates {
        // append
        let mut q1 = circuit.gates.clone();
        q1.push(*g);
        let mut c1 = CircuitSeq { gates: q1 };
        c1.canonicalize();
        if !c1.adjacent_id() {
            if let Some(canon) = c1.canonicalize_polys(DEDUP_N, true) {
                let hash: u128 = xxh3_128(&polys_repr_blob(&canon.0));
                set.insert((
                    hash.to_le_bytes().to_vec(),
                    encode_circuit(&canon.1.repr_blob()),
                ));
            }
        }
        // prepend
        let mut q2 = Vec::with_capacity(circuit.gates.len() + 1);
        q2.push(*g);
        q2.extend_from_slice(&circuit.gates);
        let mut c2 = CircuitSeq { gates: q2 };
        c2.canonicalize();
        if !c2.adjacent_id() {
            if let Some(canon) = c2.canonicalize_polys(DEDUP_N, true) {
                let hash: u128 = xxh3_128(&polys_repr_blob(&canon.0));
                set.insert((
                    hash.to_le_bytes().to_vec(),
                    encode_circuit(&canon.1.repr_blob()),
                ));
            }
        }
    }
    set
}

fn check_dedup_circuit(circuit: &CircuitSeq, min_n: usize, label: &str) -> bool {
    let u = touched_wires(circuit).len();
    let total = ordered3(DEDUP_N);

    let (old_gates, old_skip) = old_filtered(circuit, DEDUP_N, min_n, 0);
    let (new_gates, new_skip) = abstract_gates_for_circuit_filtered(circuit, DEDUP_N, min_n, 0);

    let old_ok = old_gates.len() + old_skip == total;
    let new_ok = new_gates.len() + new_skip == total;

    let old_set = candidate_set(circuit, &old_gates);
    let new_set = candidate_set(circuit, &new_gates);
    let sets_equal = old_set == new_set;

    let pass = old_ok && new_ok && sets_equal;
    println!(
        "[{}] u={} min_n={} old_emitted={} new_emitted={} old_skip={} new_skip={} old_acct={} new_acct={} unique_kv_old={} unique_kv_new={} sets_equal={} => {}",
        label,
        u,
        min_n,
        old_gates.len(),
        new_gates.len(),
        old_skip,
        new_skip,
        old_ok,
        new_ok,
        old_set.len(),
        new_set.len(),
        sets_equal,
        if pass { "PASS" } else { "FAIL" },
    );
    pass
}

#[test]
#[ignore = "requires ./rocks_db_m7 in the working directory (dbgen tree)"]
fn fresh_wire_dedup_kv_sets_match() {
    let db = open_db_for_read(DEDUP_M - 1);

    // Bucket sample circuits by used-wire count.
    let want = |u: usize| -> usize {
        match u {
            14 => 60,
            15 => 30,
            16 => 12,
            17 => 6,
            _ => 4,
        }
    };
    let mut buckets: BTreeMap<usize, Vec<CircuitSeq>> = BTreeMap::new();

    let iter = db.iterator(rocksdb::IteratorMode::Start);
    let mut rows_scanned = 0usize;
    'scan: for item in iter {
        let (_k, v) = item.expect("rocksdb iter error");
        rows_scanned += 1;
        let mut pos = 0usize;
        while pos < v.len() {
            let len = v[pos] as usize;
            pos += 1;
            if pos + len > v.len() {
                break;
            }
            let c = CircuitSeq::from_blob(&v[pos..pos + len]);
            pos += len;
            let u = touched_wires(&c).len();
            let bucket = buckets.entry(u).or_default();
            if bucket.len() < want(u) {
                bucket.push(c);
            }
        }
        if rows_scanned >= 500_000 {
            break 'scan;
        }
        if rows_scanned % 50_000 == 0 {
            let filled: usize = buckets.values().map(|b| b.len()).sum();
            let full_low = buckets.get(&14).map_or(false, |b| b.len() >= want(14))
                && buckets.get(&15).map_or(false, |b| b.len() >= want(15))
                && buckets.get(&16).map_or(false, |b| b.len() >= want(16));
            eprintln!("scanned {} rows, sampled {} circuits", rows_scanned, filled);
            if full_low && rows_scanned >= 200_000 {
                break 'scan;
            }
        }
    }

    let mut jobs: Vec<(CircuitSeq, usize, String)> = Vec::new();
    for (u, circuits) in &buckets {
        for (i, c) in circuits.iter().enumerate() {
            jobs.push((c.clone(), DEDUP_MIN_N, format!("u{}_{}", u, i)));
            // Also exercise ALL gate classes (no filter) on a few circuits per
            // bucket, so classes normally filtered at min_n=17 are validated too.
            if i < 2 {
                jobs.push((c.clone(), 0, format!("u{}_{}_min0", u, i)));
            }
        }
    }

    println!(
        "Validating {} jobs across buckets: {:?}",
        jobs.len(),
        buckets.iter().map(|(u, b)| (*u, b.len())).collect::<Vec<_>>()
    );

    let results: Vec<bool> = jobs
        .par_iter()
        .map(|(c, min_n, label)| check_dedup_circuit(c, *min_n, label))
        .collect();

    let passed = results.iter().filter(|&&p| p).count();
    let failed = results.len() - passed;
    println!("=== {} passed, {} failed ===", passed, failed);
    assert_eq!(failed, 0, "fresh-wire dedup produced divergent (key,value) sets");
}

// ======================================================= 2rocks cap validation
// Capped mapping enumeration in build_from_2rocks: for sampled (c1, c2)
// pairs, across all four concatenation cases and several min_n values, the
// surviving-mapping sequence, skip accounting, density premise, and (on a
// subset) end-to-end (key, value) outputs must match the full enumeration.

/// Exact copy of the c1_rev / c2_rev preparation from build_from_2rocks.
fn rev_canonical(c: &CircuitSeq) -> CircuitSeq {
    let mut r = CircuitSeq {
        gates: c.gates.iter().rev().cloned().collect(),
    };
    r.canonicalize();
    let used = r.used_wires();
    let wire_map: HashMap<u16, u16> = used
        .iter()
        .enumerate()
        .map(|(i, &w)| (w, i as u16))
        .collect();
    r = CircuitSeq {
        gates: r
            .gates
            .iter()
            .map(|&[t, c1, c2]| [wire_map[&t], wire_map[&c1], wire_map[&c2]])
            .collect(),
    };
    r.canonicalize();
    let n2 = r.max_wire() as usize + 1;
    let canon = canonicalize_polys_4(r.to_polynomial(n2, 0, r.gates.len()), true).unwrap();
    r.rewire(&canon.1.invert(), n2);
    r.canonicalize();
    r
}

#[derive(Default)]
struct CaseOutcome {
    surviving_mappings: Vec<Vec<u16>>,
    skipped: usize,
    density_violations: usize,
    kv: Vec<(Vec<u8>, Vec<u8>)>,
}

fn run_2rocks_case(
    first: &CircuitSeq,
    second: &CircuitSeq,
    map_first: bool,
    n_first: usize,
    n_second: usize,
    n_total: usize,
    min_n: usize,
    capped: bool,
    full_kv: bool,
) -> CaseOutcome {
    let mut out = CaseOutcome::default();

    // In build_from_2rocks the mapping always applies to the circuit whose
    // wire count is the SECOND argument of for_each_mapping.
    let (na, nb, mapped_circuit, fixed_circuit) = if map_first {
        (n_second, n_first, first, second)
    } else {
        (n_first, n_second, second, first)
    };

    let k_cap = (na + nb).saturating_sub(min_n);
    if capped {
        out.skipped += count_mappings_pruned(na, nb, k_cap);
    }

    let mut handle = |mapping: &[u16], k_opt: Option<usize>| {
        let mapped = apply_wire_mapping(mapped_circuit, mapping);
        let (first_gates, second_gates): (&[[u16; 3]], &[[u16; 3]]) = if map_first {
            (&mapped.gates, &fixed_circuit.gates)
        } else {
            (&fixed_circuit.gates, &mapped.gates)
        };
        let mut gates = Vec::with_capacity(first_gates.len() + second_gates.len());
        gates.extend_from_slice(first_gates);
        gates.extend_from_slice(second_gates);
        let mut combined = CircuitSeq { gates };
        combined.canonicalize();
        let used = combined.used_wires().len();
        if let Some(k) = k_opt {
            if used != na + nb - k {
                out.density_violations += 1;
            }
        }
        if used < min_n {
            out.skipped += 1;
            return;
        }
        out.surviving_mappings.push(mapping.to_vec());
        if combined.adjacent_id() {
            return;
        }
        if full_kv {
            let (canon_polys, canon_circuit, _, _, _) =
                combined.canonicalize_polys(n_total, true).unwrap();
            let key = xxh3_128(&polys_repr_blob(&canon_polys))
                .to_le_bytes()
                .to_vec();
            let value = encode_circuit(&canon_circuit.repr_blob());
            out.kv.push((key, value));
        }
    };

    if capped {
        for_each_mapping_capped(na, nb, k_cap, |mapping, k| handle(mapping, Some(k)));
    } else {
        for_each_mapping(na, nb, |mapping| handle(mapping, None));
    }

    out
}

#[test]
#[ignore = "requires ./rocks_db_m2 and ./rocks_db_m3 in the working directory (dbgen tree)"]
fn capped_2rocks_enumeration_matches_full() {
    let db1 = open_db_for_read(3);
    let db2 = open_db_for_read(2);

    let load = |db: &rocksdb::DB, limit: usize| -> Vec<CircuitSeq> {
        let mut out = Vec::new();
        for item in db.iterator(rocksdb::IteratorMode::Start) {
            let (_k, v) = item.expect("iter");
            let mut pos = 0usize;
            while pos < v.len() {
                let len = v[pos] as usize;
                pos += 1;
                if pos + len > v.len() {
                    break;
                }
                out.push(CircuitSeq::from_blob(&v[pos..pos + len]));
                pos += len;
                if out.len() >= limit {
                    return out;
                }
            }
        }
        out
    };

    let c1s = load(&db1, 12);
    let c2s = load(&db2, 6);
    println!(
        "Sampled {} c1 (m3) and {} c2 (m2) circuits",
        c1s.len(),
        c2s.len()
    );

    // (c1 idx, c2 idx, min_n, full_kv). min_n values exercise: no pruning (0),
    // moderate, aggressive, and impossible (everything pruned). Full
    // (key,value) end-to-end comparison runs on a subset at aggressive min_n
    // where the surviving-mapping count is small.
    let mut jobs: Vec<(usize, usize, usize, bool)> = Vec::new();
    for i in 0..c1s.len() {
        for j in 0..c2s.len() {
            for &min_n in &[0usize, 8, 11, 13, 40] {
                let full_kv = min_n >= 11 && i < 4 && j < 3;
                jobs.push((i, j, min_n, full_kv));
            }
        }
    }
    println!("Running {} pair validations", jobs.len());

    let m_total = 3 + 2;
    let n_total = 3 * m_total;

    let failures: usize = jobs
        .par_iter()
        .map(|&(i, j, min_n, full_kv)| {
            let c1 = &c1s[i];
            let c2 = &c2s[j];
            let n1 = touched_wires(c1).len();
            let n2 = touched_wires(c2).len();
            let c1_rev = rev_canonical(c1);
            let c2_rev = rev_canonical(c2);
            let n1_rev = touched_wires(&c1_rev).len();
            let n2_rev = touched_wires(&c2_rev).len();

            let cases: [(&CircuitSeq, &CircuitSeq, bool, usize, usize); 4] = [
                (c1, c2, false, n1, n2),          // case 1: c1 || mapped_c2
                (c2, c1, false, n2, n1),          // case 2: c2 || mapped_c1
                (&c1_rev, c2, false, n1_rev, n2), // case 3: c1_rev || mapped_c2
                (c1, &c2_rev, true, n1, n2_rev),  // case 4: mapped_c1 || c2_rev
            ];

            let mut fail = 0usize;
            for (ci, &(first, second, map_first, nf, ns)) in cases.iter().enumerate() {
                let old = run_2rocks_case(
                    first, second, map_first, nf, ns, n_total, min_n, false, full_kv,
                );
                let new = run_2rocks_case(
                    first, second, map_first, nf, ns, n_total, min_n, true, full_kv,
                );
                let ok = old.surviving_mappings == new.surviving_mappings
                    && old.skipped == new.skipped
                    && new.density_violations == 0
                    && old.kv == new.kv;
                if !ok {
                    fail += 1;
                    println!(
                        "FAIL pair=({},{}) case={} min_n={} nf={} ns={} old_surv={} new_surv={} old_skip={} new_skip={} density_violations={} kv_equal={}",
                        i,
                        j,
                        ci + 1,
                        min_n,
                        nf,
                        ns,
                        old.surviving_mappings.len(),
                        new.surviving_mappings.len(),
                        old.skipped,
                        new.skipped,
                        new.density_violations,
                        old.kv == new.kv,
                    );
                }
            }
            fail
        })
        .sum();

    assert_eq!(
        failures, 0,
        "capped 2rocks enumeration diverged from the full enumeration"
    );
    println!("VALIDATION PASSED ({} jobs, all 4 cases each)", jobs.len());
}

// ===================================================== merge prefix validation
// Independent ground-truth check for merge_rocks_parallel over one 2-byte key
// prefix: rebuild the correct merge directly from the sources and compare
// order-independent digests against the parallel binary's output.

fn parse_blobs(v: &[u8], out: &mut Vec<Vec<u8>>) {
    let mut pos = 0usize;
    while pos + 1 <= v.len() {
        let len = v[pos] as usize;
        pos += 1;
        if pos + len > v.len() {
            break;
        }
        out.push(v[pos..pos + len].to_vec());
        pos += len;
    }
}

fn digest_set(k: &[u8], blobs: &BTreeSet<Vec<u8>>) -> u128 {
    let mut buf = Vec::with_capacity(k.len() + 4);
    buf.extend_from_slice(k);
    buf.push(0xff);
    for b in blobs {
        buf.push(b.len() as u8);
        buf.extend_from_slice(b);
    }
    xxh3_128(&buf)
}

fn fold_digests(map: &HashMap<Vec<u8>, BTreeSet<Vec<u8>>>) -> (u64, u128, u128) {
    let (mut n, mut x, mut s) = (0u64, 0u128, 0u128);
    for (k, set) in map {
        let h = digest_set(k, set);
        n += 1;
        x ^= h;
        s = s.wrapping_add(h);
    }
    (n, x, s)
}

#[test]
#[ignore = "requires MERGE_PAR_DB, MERGE_SOURCES (colon-separated), MERGE_TEST_PREFIX (2 hex bytes)"]
fn merge_prefix_rebuild_matches() {
    use rocksdb::{DB, Direction, IteratorMode, Options, ReadOptions};

    let par_db = std::env::var("MERGE_PAR_DB").expect("MERGE_PAR_DB required");
    let sources: Vec<String> = std::env::var("MERGE_SOURCES")
        .expect("MERGE_SOURCES required (colon-separated)")
        .split(':')
        .map(String::from)
        .collect();
    let pfx = std::env::var("MERGE_TEST_PREFIX").expect("MERGE_TEST_PREFIX required");
    let b0 = u8::from_str_radix(&pfx[0..2], 16).unwrap();
    let b1 = u8::from_str_radix(&pfx[2..4], 16).unwrap();
    let mut lower = vec![0u8; 16];
    lower[0] = b0;
    lower[1] = b1;
    let mut upper = vec![0u8; 16];
    if b1 == 0xff {
        upper[0] = b0 + 1;
    } else {
        upper[0] = b0;
        upper[1] = b1 + 1;
    }

    // Ground truth straight from the sources.
    let ropts = {
        let mut o = Options::default();
        o.set_max_open_files(-1);
        o
    };
    let mut truth: HashMap<Vec<u8>, BTreeSet<Vec<u8>>> = HashMap::new();
    for src in &sources {
        let db = DB::open_for_read_only(&ropts, src, false).expect("open source");
        let mut ro = ReadOptions::default();
        ro.set_total_order_seek(true);
        ro.set_iterate_upper_bound(upper.clone());
        for item in db.iterator_opt(IteratorMode::From(&lower, Direction::Forward), ro) {
            let (k, v) = item.expect("iter");
            let mut blobs = Vec::new();
            parse_blobs(&v, &mut blobs);
            let e = truth.entry(k.to_vec()).or_default();
            for b in blobs {
                e.insert(b);
            }
        }
    }
    let truth_d = fold_digests(&truth);

    // Parallel output (holds only prefix keys).
    let db = DB::open_for_read_only(&ropts, &par_db, false).expect("open par db");
    let mut got: HashMap<Vec<u8>, BTreeSet<Vec<u8>>> = HashMap::new();
    for item in db.iterator(IteratorMode::Start) {
        let (k, v) = item.expect("iter");
        let mut blobs = Vec::new();
        parse_blobs(&v, &mut blobs);
        let e = got.entry(k.to_vec()).or_default();
        for b in blobs {
            e.insert(b);
        }
    }
    let got_d = fold_digests(&got);

    println!(
        "truth: keys={} xor={:032x} sum={:032x}",
        truth_d.0, truth_d.1, truth_d.2
    );
    println!(
        "par:   keys={} xor={:032x} sum={:032x}",
        got_d.0, got_d.1, got_d.2
    );
    assert_eq!(truth_d, got_d, "parallel merge diverged from ground truth");
    println!("MERGE_VERIFY_MATCH");
}
