//! Build per-permutation circuit DBs (the SGDB construction generalized) for
//! every curated permutation whose minimal retained friend has min..=max
//! gates (default M2..M3).
//!
//! Per sample for target permutation P:
//!   1. F = a random friend of P's own entry (computes P in the entry frame);
//!   2. I = reverse(C) ++ C' from a random curated friend pair (C a 7-gate
//!      friend, C' another friend of the same entry) — an identity, exactly
//!      the SGDB raw material — cyclically rotated at a random point
//!      (rotations of an identity are identities);
//!   3. I's wires are mapped injectively at random into F's wire range plus
//!      head-room (an identity is function-preserving under ANY placement);
//!   4. the mapped identity is inserted at a random position inside F;
//!   5. VERIFY against the entry's minimal friend on the union of used wires
//!      (exhaustive <=16 wires, 4096 random states above), dedup, keep.
//!
//! Output: <out>/m{K}_{seq:05}.sgdb1 per permutation (same line format as
//! SGDB: header `sgdb1 N`, then `t,p,n;...` per circuit) + MANIFEST.tsv
//! mapping file -> min size, minimal spelling, friend count, kept count.
//!
//! Width restrictions (the SGDB conventions): a kept circuit must SPAN
//! min-span..=max-span distinct wires (defaults 7..=12 — the >=7 floor from
//! the span7 SGDB, and a cap that keeps the circuits store-digestible). The
//! identity is mapped into a narrow window (max(F_top+1, 13) wires) so the
//! cap is reachable by overlap rather than luck.
//!
//! Usage: mgdb_build <curated_dir> [--count N] [--min 2] [--max 3]
//!                   [--min-span 7] [--max-span 12] [--out DIR] [--seed S]
//!                   [--target-lens A,B]
//!
//! --target-lens L1,L2 switches to the SHORT-IDENTITY construction: each
//! sample is the target's MINIMAL friend with a rotated, wire-mapped identity
//! reverse(A)++B inserted, where (A, B) is a same-entry friend pair harvested
//! with |A|+|B| = T - |minimal| for a random T in {L1, L2}. This reaches
//! total lengths the 7-gate-C construction cannot (e.g. all-10/11-gate
//! pools). Identity pair lengths outside what the store offers make that T
//! unavailable; the build reports the realized length histogram.

use local_mixing::replace::frozen::scan_shard;
use rand::rngs::StdRng;
use rand::seq::SliceRandom;
use rand::{Rng, SeedableRng};
use std::collections::HashSet;
use std::io::Write;

type G57 = [u16; 3];

fn parse_friends(value: &[u8], swap: bool) -> Vec<Vec<G57>> {
    let mut friends = Vec::new();
    let mut pos = 0usize;
    while pos < value.len() {
        let len = value[pos] as usize;
        pos += 1;
        if len == 0 || len % 3 != 0 || pos + len > value.len() {
            return Vec::new();
        }
        friends.push(
            value[pos..pos + len]
                .chunks(3)
                .map(|c| {
                    if swap {
                        [c[0] as u16, c[2] as u16, c[1] as u16]
                    } else {
                        [c[0] as u16, c[1] as u16, c[2] as u16]
                    }
                })
                .collect(),
        );
        pos += len;
    }
    friends
}

fn apply(gates: &[G57], mut x: u64) -> u64 {
    for &[t, p, n] in gates {
        let fire = ((x >> p) & 1) | (1 ^ ((x >> n) & 1));
        x ^= fire << t;
    }
    x
}

fn max_wire(gates: &[G57]) -> u16 {
    gates.iter().flat_map(|g| g.iter().copied()).max().unwrap_or(0)
}

/// candidate == reference as functions over `nw` wires?
fn equivalent(cand: &[G57], reference: &[G57], nw: usize, rng: &mut StdRng) -> bool {
    if nw <= 16 {
        for x in 0u64..(1u64 << nw) {
            if apply(cand, x) != apply(reference, x) {
                return false;
            }
        }
        true
    } else {
        let mask = if nw >= 64 { u64::MAX } else { (1u64 << nw) - 1 };
        (0..4096).all(|_| {
            let x = rng.random::<u64>() & mask;
            apply(cand, x) == apply(reference, x)
        })
    }
}

fn main() {
    let mut a = std::env::args().skip(1);
    let dir = a.next().expect("usage: mgdb_build <curated_dir> [--count N] [--min A] [--max B] [--out D] [--seed S]");
    let (mut count, mut lo, mut hi) = (1000usize, 2usize, 3usize);
    let (mut span_lo, mut span_hi) = (7usize, 12usize);
    let (mut out_dir, mut seed) = ("mgdb_out".to_string(), 20260815u64);
    let mut target_lens: Vec<usize> = Vec::new();
    // > 0: exact per-size quotas — every target gets per_size circuits at
    // EVERY length in --target-lens (count is then ignored).
    let mut per_size = 0usize;
    while let Some(arg) = a.next() {
        let mut v = || a.next().expect("missing value");
        match arg.as_str() {
            "--count" => count = v().parse().expect("bad count"),
            "--min" => lo = v().parse().expect("bad min"),
            "--max" => hi = v().parse().expect("bad max"),
            "--min-span" => span_lo = v().parse().expect("bad min-span"),
            "--max-span" => span_hi = v().parse().expect("bad max-span"),
            "--out" => out_dir = v(),
            "--seed" => seed = v().parse().expect("bad seed"),
            "--target-lens" => {
                target_lens = v()
                    .split(',')
                    .map(|x| x.parse().expect("bad target len"))
                    .collect()
            }
            "--count-per-size" => per_size = v().parse().expect("bad count-per-size"),
            _ => panic!("unknown arg {arg}"),
        }
    }
    std::fs::create_dir_all(&out_dir).expect("create out dir");
    let mut rng = StdRng::seed_from_u64(seed);

    // Full scan: collect every target entry (min friend size in lo..=hi), a
    // reservoir of 7-gate-C identity material (the long construction), and —
    // when --target-lens is set — reservoirs of SHORT same-entry friend
    // pairs keyed by |A|+|B| (the short-identity construction).
    const POOL_CAP: usize = 120_000;
    const SHORT_CAP: usize = 80_000;
    let needed_ls: Vec<usize> = {
        let mut v: Vec<usize> = target_lens
            .iter()
            .flat_map(|&t| (lo..=hi).filter_map(move |mn| t.checked_sub(mn)))
            .filter(|&l| l >= 2)
            .collect();
        v.sort_unstable();
        v.dedup();
        v
    };
    let mut targets: Vec<Vec<Vec<G57>>> = Vec::new();
    let mut pool: Vec<Vec<Vec<G57>>> = Vec::with_capacity(POOL_CAP);
    let mut pool_seen = 0u64;
    let mut short_pool: std::collections::HashMap<usize, Vec<(Vec<G57>, Vec<G57>)>> =
        needed_ls.iter().map(|&l| (l, Vec::new())).collect();
    let mut short_seen: std::collections::HashMap<usize, u64> =
        needed_ls.iter().map(|&l| (l, 0u64)).collect();
    for s in 0..256usize {
        scan_shard(&dir, s, &mut |value: &[u8]| {
            let fr = parse_friends(value, true); // legacy-swapped-controls
            if fr.is_empty() {
                return;
            }
            let mn = fr.iter().map(|f| f.len()).min().unwrap();
            if mn >= lo && mn <= hi {
                targets.push(fr.clone());
            }
            if fr.len() >= 2 && fr.iter().any(|f| f.len() == 7) {
                pool_seen += 1;
                if pool.len() < POOL_CAP {
                    pool.push(fr.clone());
                } else {
                    let j = rng.random_range(0..pool_seen);
                    if (j as usize) < POOL_CAP {
                        pool[j as usize] = fr.clone();
                    }
                }
            }
            // Short-pair harvest: up to 4 qualifying same-entry pairs per
            // needed length per entry, reservoir-merged across the store.
            if !needed_ls.is_empty() && fr.len() >= 2 {
                for &l in &needed_ls {
                    let mut found = 0;
                    'pairs: for i in 0..fr.len() {
                        for j in 0..fr.len() {
                            if i == j || fr[i].len() + fr[j].len() != l {
                                continue;
                            }
                            let seen = short_seen.get_mut(&l).unwrap();
                            *seen += 1;
                            let p = short_pool.get_mut(&l).unwrap();
                            if p.len() < SHORT_CAP {
                                p.push((fr[i].clone(), fr[j].clone()));
                            } else {
                                let k = rng.random_range(0..*seen);
                                if (k as usize) < SHORT_CAP {
                                    p[k as usize] = (fr[i].clone(), fr[j].clone());
                                }
                            }
                            found += 1;
                            if found >= 4 {
                                break 'pairs;
                            }
                        }
                    }
                }
            }
        });
        if s % 64 == 63 {
            eprintln!("[mgdb] shard {s:02x}: {} targets, pool {}", targets.len(), pool.len());
        }
    }
    eprintln!("[mgdb] scan done: {} target permutations, identity pool {} entries", targets.len(), pool.len());
    for &l in &needed_ls {
        eprintln!("[mgdb] short-identity pairs of length {l}: {}", short_pool[&l].len());
    }
    assert!(!pool.is_empty(), "empty identity pool");

    let mut manifest = std::fs::File::create(format!("{out_dir}/MANIFEST.tsv")).expect("manifest");
    writeln!(manifest, "file\tmin_gates\tstore_friends\tkept\tminimal_circuit").unwrap();
    let (mut total_kept, mut rej_verify, mut rej_dup, mut rej_wide) = (0u64, 0u64, 0u64, 0u64);
    let mut len_hist: std::collections::BTreeMap<usize, u64> = std::collections::BTreeMap::new();
    let mut underfilled = 0u64;

    for (idx, fr) in targets.iter().enumerate() {
        let mn = fr.iter().map(|f| f.len()).min().unwrap();
        let reference = fr.iter().filter(|f| f.len() == mn).min_by_key(|f| f.to_vec()).unwrap().clone();
        let want_total = if per_size > 0 { per_size * target_lens.len() } else { count };
        let mut quota: std::collections::HashMap<usize, usize> = if per_size > 0 {
            target_lens.iter().map(|&t| (t, per_size)).collect()
        } else {
            std::collections::HashMap::new()
        };
        let mut kept: Vec<Vec<G57>> = Vec::with_capacity(want_total);
        let mut seen: HashSet<Vec<G57>> = HashSet::new();
        let mut attempts = 0u64;
        while kept.len() < want_total && attempts < 40 * want_total as u64 {
            attempts += 1;
            let (f, ident): (&Vec<G57>, Vec<G57>) = if target_lens.is_empty() {
                // long construction: any friend + a 7-gate-C identity
                let f = &fr[rng.random_range(0..fr.len())];
                let e = &pool[rng.random_range(0..pool.len())];
                let sevens: Vec<usize> = (0..e.len()).filter(|&i| e[i].len() == 7).collect();
                let ci = sevens[rng.random_range(0..sevens.len())];
                let others: Vec<usize> = (0..e.len()).filter(|&i| i != ci).collect();
                let cpi = others[rng.random_range(0..others.len())];
                let mut ident: Vec<G57> = e[ci].iter().rev().copied().collect();
                ident.extend_from_slice(&e[cpi]);
                (f, ident)
            } else {
                // short construction: the MINIMAL friend + a same-entry pair
                // identity of exactly T - |minimal| gates. Quota mode draws T
                // among the lengths still owed.
                let t = if per_size > 0 {
                    let open: Vec<usize> =
                        target_lens.iter().copied().filter(|l| quota[l] > 0).collect();
                    if open.is_empty() {
                        break;
                    }
                    open[rng.random_range(0..open.len())]
                } else {
                    target_lens[rng.random_range(0..target_lens.len())]
                };
                let Some(l) = t.checked_sub(reference.len()).filter(|&l| l >= 2) else {
                    continue;
                };
                let Some(p) = short_pool.get(&l).filter(|p| !p.is_empty()) else {
                    continue;
                };
                let (a, b) = &p[rng.random_range(0..p.len())];
                let mut ident: Vec<G57> = a.iter().rev().copied().collect();
                ident.extend_from_slice(b);
                (&reference, ident)
            };
            let ident = ident;
            let k = rng.random_range(0..ident.len());
            let ident: Vec<G57> = ident[k..].iter().chain(ident[..k].iter()).copied().collect();
            // random injective wire map into a NARROW window over F's range,
            // so the span cap is reachable by overlap rather than luck
            let f_top = max_wire(f) + 1;
            let space = (f_top + 1).max(span_hi as u16 + 1);
            let mut iw: Vec<u16> = ident.iter().flat_map(|g| g.iter().copied()).collect();
            iw.sort_unstable();
            iw.dedup();
            if iw.len() as u16 > space {
                rej_wide += 1;
                continue;
            }
            let mut slots: Vec<u16> = (0..space).collect();
            slots.shuffle(&mut rng);
            let map: std::collections::HashMap<u16, u16> =
                iw.iter().copied().zip(slots).collect();
            let ident_m: Vec<G57> =
                ident.iter().map(|&[t, p, n]| [map[&t], map[&p], map[&n]]).collect();
            // insert at a random seam in F
            let j = rng.random_range(0..=f.len());
            let mut cand: Vec<G57> = Vec::with_capacity(f.len() + ident_m.len());
            cand.extend_from_slice(&f[..j]);
            cand.extend_from_slice(&ident_m);
            cand.extend_from_slice(&f[j..]);
            let mut used: Vec<u16> = cand.iter().flat_map(|g| g.iter().copied()).collect();
            used.sort_unstable();
            used.dedup();
            if used.len() < span_lo || used.len() > span_hi {
                rej_wide += 1;
                continue;
            }
            let nw = max_wire(&cand).max(max_wire(&reference)) as usize + 1;
            if nw > 40 {
                rej_wide += 1;
                continue;
            }
            if !equivalent(&cand, &reference, nw, &mut rng) {
                rej_verify += 1;
                continue;
            }
            if !seen.insert(cand.clone()) {
                rej_dup += 1;
                continue;
            }
            *len_hist.entry(cand.len()).or_insert(0) += 1;
            if per_size > 0 {
                *quota.get_mut(&cand.len()).expect("kept length outside target lens") -= 1;
            }
            kept.push(cand);
        }
        let fname = format!("m{mn}_{idx:05}.sgdb1");
        let mut fh = std::fs::File::create(format!("{out_dir}/{fname}")).expect("db file");
        writeln!(fh, "sgdb1 {}", kept.len()).unwrap();
        for c in &kept {
            let line: Vec<String> = c.iter().map(|&[t, p, n]| format!("{t},{p},{n}")).collect();
            writeln!(fh, "{}", line.join(";")).unwrap();
        }
        let minc: Vec<String> = reference.iter().map(|&[t, p, n]| format!("{t},{p},{n}")).collect();
        writeln!(manifest, "{fname}\t{mn}\t{}\t{}\t{}", fr.len(), kept.len(), minc.join(";")).unwrap();
        total_kept += kept.len() as u64;
        if kept.len() < want_total {
            underfilled += 1;
        }
        if idx % 200 == 199 {
            eprintln!("[mgdb] {} / {} permutations done", idx + 1, targets.len());
        }
    }
    let lh: Vec<String> = len_hist.iter().map(|(l, c)| format!("{l}g:{c}")).collect();
    println!("[mgdb] length histogram: {}", lh.join(" "));
    println!("[mgdb] underfilled targets (kept < count): {underfilled}");
    println!(
        "[mgdb] wrote {} circuits across {} permutation DBs in {out_dir}; rejects: verify={rej_verify} dup={rej_dup} wide={rej_wide}",
        total_kept,
        targets.len()
    );
    assert_eq!(rej_verify, 0, "verification rejects should be impossible — investigate before using these DBs");
}
