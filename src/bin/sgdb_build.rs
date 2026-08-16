//! Build a single-gate DB (SGDB): circuits that compute THE canonical
//! single-g57-gate permutation (wire0 ^= wire1 OR NOT wire2), harvested from
//! the curated store.
//!
//! Construction per sample:
//!   1. pick a curated entry with >=2 friends, one of which has 7 gates;
//!   2. C = a random 7-gate friend, C' = a random OTHER friend;
//!   3. I = reverse(C) ++ C'  — an identity (g57 gates are involutions);
//!   4. remove a random gate g at position k and ROTATE: out = I[k+1..] ++ I[..k].
//!      For an identity A.g.B the rotation B.A computes exactly g (removing g
//!      without rotating would give A.g.A^-1, a conjugate — not g itself);
//!   5. relabel wires so g = (0,1,2) (target, pos, neg), scratch wires 3.. in
//!      first-use order;
//!   6. VERIFY by exhaustive truth table over the used wires, dedup, keep.
//!
//! The curated value convention (control swap) is auto-detected by checking
//! which decode makes reverse(C)++C' an identity.
//!
//! Output: text; header "sgdb1 <count>", then one circuit per line as
//! ';'-joined "t,p,n" g57 triples in canonical wire space.
//!
//! Usage: sgdb_build <curated_dir> [--count N] [--out FILE] [--seed S]
//!                    [--min-span W]
//!
//! --min-span W rejects circuits spanning fewer than W distinct wires
//! (default 5, i.e. spans <=4 are rejected and counted); the kept-circuit
//! span histogram is printed at the end so a final DB can be sub-sampled
//! from a larger pool by span.

use local_mixing::replace::frozen::scan_shard;
use rand::rngs::StdRng;
use rand::seq::SliceRandom;
use rand::{Rng, SeedableRng};
use std::collections::HashSet;
use std::io::Write;

type G57 = [u16; 3]; // (target, pos, neg): t ^= p OR NOT n

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

fn used_wires(gates: &[G57]) -> Vec<u16> {
    let mut ws: Vec<u16> = gates.iter().flat_map(|g| g.iter().copied()).collect();
    ws.sort_unstable();
    ws.dedup();
    ws
}

/// Exhaustive check over the circuit's used wires (relabeled densely) that
/// `gates` computes `expected` (also over the same dense space).
fn equals_fn(gates: &[G57], expected: impl Fn(u64) -> u64, nw: usize) -> bool {
    if nw > 22 {
        return false; // caller counts this as a width reject
    }
    for x in 0u64..(1u64 << nw) {
        if apply(gates, x) != expected(x) {
            return false;
        }
    }
    true
}

fn densify(gates: &[G57]) -> (Vec<G57>, usize) {
    let ws = used_wires(gates);
    let map = |w: u16| ws.binary_search(&w).unwrap() as u16;
    (gates.iter().map(|&[t, p, n]| [map(t), map(p), map(n)]).collect(), ws.len())
}

fn main() {
    let mut a = std::env::args().skip(1);
    let dir = a.next().expect("usage: sgdb_build <curated_dir> [--count N] [--out F] [--seed S]");
    let (mut count, mut out_path, mut seed) = (10_000usize, "sgdb.sgdb1".to_string(), 20260813u64);
    let mut min_span = 5usize;
    let (mut len_lo, mut len_hi) = (0usize, usize::MAX);
    while let Some(arg) = a.next() {
        let mut v = || a.next().expect("missing value");
        match arg.as_str() {
            "--count" => count = v().parse().expect("bad count"),
            "--out" => out_path = v(),
            "--seed" => seed = v().parse().expect("bad seed"),
            "--min-span" => min_span = v().parse().expect("bad min-span"),
            "--len-min" => len_lo = v().parse().expect("bad len-min"),
            "--len-max" => len_hi = v().parse().expect("bad len-max"),
            _ => panic!("unknown arg {arg}"),
        }
    }
    let mut rng = StdRng::seed_from_u64(seed);
    let mut shards: Vec<usize> = (0..256).collect();
    shards.shuffle(&mut rng);

    // -------- convention detection on the first shard: which decode makes
    // reverse(C) ++ C' an identity?
    let mut swap_votes = (0u32, 0u32); // (swapped ok, native ok)
    scan_shard(&dir, shards[0], &mut |value: &[u8]| {
        if swap_votes.0 + swap_votes.1 >= 200 {
            return;
        }
        for (swap, vote) in [(true, 0usize), (false, 1usize)] {
            let fr = parse_friends(value, swap);
            if fr.len() < 2 {
                continue;
            }
            let mut ident: Vec<G57> = fr[0].iter().rev().copied().collect();
            ident.extend_from_slice(&fr[1]);
            let (dense, nw) = densify(&ident);
            if nw <= 22 && equals_fn(&dense, |x| x, nw) {
                if vote == 0 {
                    swap_votes.0 += 1
                } else {
                    swap_votes.1 += 1
                }
            }
        }
    });
    let swap = swap_votes.0 >= swap_votes.1;
    eprintln!(
        "[sgdb] convention votes: swapped={} native={} -> using {}",
        swap_votes.0,
        swap_votes.1,
        if swap { "legacy-swapped-controls" } else { "native" }
    );
    assert!(
        swap_votes.0.max(swap_votes.1) > 20,
        "convention detection failed — too few identity-verified pairs"
    );

    let mut kept: Vec<Vec<G57>> = Vec::new();
    let mut seen: HashSet<Vec<G57>> = HashSet::new();
    let (mut rej_verify, mut rej_wide, mut rej_dup, mut rej_deg) = (0u64, 0u64, 0u64, 0u64);
    let mut rej_narrow = 0u64;
    let mut span_hist: std::collections::BTreeMap<usize, u64> = std::collections::BTreeMap::new();

    'shards: for &s in &shards {
        let mut entries: Vec<Vec<Vec<G57>>> = Vec::new();
        scan_shard(&dir, s, &mut |value: &[u8]| {
            let fr = parse_friends(value, swap);
            if fr.len() >= 2 && fr.iter().any(|f| f.len() == 7) {
                entries.push(fr);
            }
        });
        entries.shuffle(&mut rng);
        for fr in entries {
            if kept.len() >= count {
                break 'shards;
            }
            let sevens: Vec<usize> =
                (0..fr.len()).filter(|&i| fr[i].len() == 7).collect();
            let ci = sevens[rng.random_range(0..sevens.len())];
            let others: Vec<usize> = (0..fr.len()).filter(|&i| i != ci).collect();
            let cpi = others[rng.random_range(0..others.len())];
            let (c, cp) = (&fr[ci], &fr[cpi]);
            let mut ident: Vec<G57> = c.iter().rev().copied().collect();
            ident.extend_from_slice(cp);
            let k = rng.random_range(0..ident.len());
            let g = ident[k];
            if g[0] == g[1] || g[0] == g[2] || g[1] == g[2] {
                rej_deg += 1;
                continue; // degenerate removed gate; skip
            }
            // rotation: suffix ++ prefix computes exactly g
            let mut cir: Vec<G57> = ident[k + 1..].to_vec();
            cir.extend_from_slice(&ident[..k]);
            // canonical relabel: g -> (0,1,2), scratch by first use
            let mut map = std::collections::HashMap::new();
            map.insert(g[0], 0u16);
            map.insert(g[1], 1u16);
            map.insert(g[2], 2u16);
            let mut next = 3u16;
            let canon: Vec<G57> = cir
                .iter()
                .map(|&[t, p, n]| {
                    let mut m = |w: u16| {
                        *map.entry(w).or_insert_with(|| {
                            let v = next;
                            next += 1;
                            v
                        })
                    };
                    [m(t), m(p), m(n)]
                })
                .collect();
            if canon.len() < len_lo || canon.len() > len_hi {
                rej_deg += 1; // counted with degenerates: out-of-length draws
                continue;
            }
            let nw = next as usize;
            if nw < min_span {
                rej_narrow += 1;
                continue;
            }
            if nw > 22 {
                rej_wide += 1;
                continue;
            }
            // verify: computes bit0 ^= bit1 OR NOT bit2, identity elsewhere
            let ok = equals_fn(&canon, |x| {
                let fire = ((x >> 1) & 1) | (1 ^ ((x >> 2) & 1));
                x ^ fire
            }, nw);
            if !ok {
                rej_verify += 1;
                continue;
            }
            if !seen.insert(canon.clone()) {
                rej_dup += 1;
                continue;
            }
            *span_hist.entry(nw).or_insert(0) += 1;
            kept.push(canon);
        }
        eprintln!("[sgdb] shard {s:02x} done — kept {}", kept.len());
    }

    let mut f = std::fs::File::create(&out_path).expect("create out");
    writeln!(f, "sgdb1 {}", kept.len()).unwrap();
    for cir in &kept {
        let line: Vec<String> =
            cir.iter().map(|&[t, p, n]| format!("{t},{p},{n}")).collect();
        writeln!(f, "{}", line.join(";")).unwrap();
    }
    let sizes: Vec<usize> = kept.iter().map(|c| c.len()).collect();
    let (mn, mx) = (sizes.iter().min().unwrap_or(&0), sizes.iter().max().unwrap_or(&0));
    println!(
        "[sgdb] wrote {} circuits to {out_path} (gates {mn}..{mx}); rejects: narrow(span<{min_span})={rej_narrow} verify={rej_verify} wide={rej_wide} dup={rej_dup} degenerate={rej_deg}",
        kept.len()
    );
    println!("[sgdb] span histogram of kept circuits:");
    for (s, c) in &span_hist {
        println!("[sgdb]   span {s:>2}: {c}");
    }
    assert_eq!(rej_verify, 0, "verification rejects should be impossible — investigate before using this DB");
}
