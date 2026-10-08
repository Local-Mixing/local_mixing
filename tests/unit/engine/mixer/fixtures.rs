// Shared fixtures for piecewise rounds and serial walks.

fn piece_cfg(pieces: usize, sequential: bool, threads: usize) -> PieceCfg {
    PieceCfg {
        pieces,
        sequential,
        threads,
        ..PieceCfg::default()
    }
}

fn meta_snapshot(mx: &Mixer) -> Vec<(u32, bool, u32, u64, u16)> {
    mx.arena
        .ids_in_order()
        .iter()
        .map(|&id| {
            let m = mx.meta_of(id);
            (m.origin, m.dir == Dir::R, m.dgen, m.litter, m.litter_size)
        })
        .collect()
}

fn rand_gate(rng: &mut StdRng, wires: u16, max_w: usize, allow_comp: bool) -> XGate {
    loop {
        let target = rng.random_range(0..wires);
        let w = rng.random_range(0..=max_w);
        let lits: Vec<(u16, bool)> = (0..w)
            .map(|_| (rng.random_range(0..wires), rng.random_bool(0.5)))
            .filter(|&(cw, _)| cw != target)
            .collect();
        if let Some(mut g) = XGate::conj(target, lits) {
            if allow_comp && g.width() == 2 && rng.random_bool(0.3) {
                g.comp = true;
            }
            return g;
        }
    }
}

fn random_g57_circuit(seed: u64, wires: u16, gates: usize) -> Vec<XGate> {
    let mut rng = StdRng::seed_from_u64(seed);
    (0..gates)
        .map(|_| {
            loop {
                let a = rng.random_range(0..wires);
                let x = rng.random_range(0..wires);
                let y = rng.random_range(0..wires);
                if a != x && a != y && x != y {
                    break XGate::from_g57([a, x, y]);
                }
            }
        })
        .collect()
}

// A conjunction-dominated circuit like real circuit_mixer input (fsplit output is
// ~90% eroded); a few g57 fossils sprinkled in.
fn random_mixed_circuit(seed: u64, wires: u16, gates: usize) -> Vec<XGate> {
    let mut rng = StdRng::seed_from_u64(seed);
    (0..gates)
        .map(|i| {
            if i % 10 == 0 {
                loop {
                    let a = rng.random_range(0..wires);
                    let x = rng.random_range(0..wires);
                    let y = rng.random_range(0..wires);
                    if a != x && a != y && x != y {
                        break XGate::from_g57([a, x, y]);
                    }
                }
            } else {
                loop {
                    let g = rand_gate(&mut rng, wires, 3, false);
                    if g.width() >= 1 {
                        break g;
                    }
                }
            }
        })
        .collect()
}

// ---- the in-memory g57 fixture store ----
//
// Every g57 circuit of at most `max_len` gates on at most `wires` wires,
// grouped by function and encoded exactly as the store builder encodes
// friends: each circuit is relabeled into the canonical wire space of its OWN
// canonicalization (forward under the forward key, reversed under the
// reverse key), and identity classes also hold the empty friend. Circuits on
// the dense wires 0..m cover every canonical shape, so any window of up to
// `max_len` g57 gates touching at most `wires` wires hits -- on any circuit
// width. Lets tests drive the stage-3 DB path (p_db = 1.0) with real
// splices, the path that replaced the historic crossing walk as the
// engine's driver.
fn g57_fixture_build(wires: u16, max_len: usize) -> std::collections::HashMap<[u8; 16], Vec<u8>> {
    use crate::canonicalization::polys_repr_blob;
    use crate::canonicalization::xgate::canonicalize_xgates_single_capped;
    use std::collections::{BTreeSet, HashMap};
    let w_max: u16 = wires;
    let mut triples: Vec<[u16; 3]> = Vec::new();
    for t in 0..w_max {
        for a in 0..w_max {
            for b in 0..w_max {
                if t != a && t != b && a != b {
                    triples.push([t, a, b]);
                }
            }
        }
    }
    let mut circuits: Vec<Vec<[u16; 3]>> = triples.iter().map(|&g| vec![g]).collect();
    let mut last = circuits.clone();
    for _ in 1..max_len {
        let mut next = Vec::new();
        for c in &last {
            for &h in &triples {
                let mut c2 = c.clone();
                c2.push(h);
                next.push(c2);
            }
        }
        circuits.extend(next.iter().cloned());
        last = next;
    }
    let mut classes: HashMap<[u8; 16], BTreeSet<Vec<u8>>> = HashMap::new();
    for c in circuits {
        // Dense wires only: every canonical shape has a dense member.
        let mut used: Vec<u16> = c.iter().flatten().copied().collect();
        used.sort_unstable();
        used.dedup();
        if used.iter().enumerate().any(|(i, &w)| i as u16 != w) {
            continue;
        }
        let gates: Vec<XGate> = c.iter().map(|&g| XGate::from_g57(g)).collect();
        let mut fwd_key = None;
        for reversed in [false, true] {
            let canon = canonicalize_xgates_single_capped(&gates, reversed, Default::default(), 0)
                .expect("fixture circuits canonicalize");
            let key = xxhash_rust::xxh3::xxh3_128(&polys_repr_blob(&canon.polys)).to_le_bytes();
            if reversed && fwd_key == Some(key) {
                continue; // the runtime decodes an equal key forward
            }
            // global -> dense (used_wires) -> canonical (inverse of order).
            let dense = |w: u16| {
                canon
                    .used_wires
                    .iter()
                    .position(|&u| u == w)
                    .expect("used wire")
            };
            let canonical = |w: u16| {
                canon
                    .order
                    .data
                    .iter()
                    .position(|&d| d == dense(w))
                    .expect("dense wire in order") as u8
            };
            let mut enc: Vec<[u8; 3]> = c
                .iter()
                .map(|g| [canonical(g[0]), canonical(g[1]), canonical(g[2])])
                .collect();
            if reversed {
                enc.reverse();
            }
            classes
                .entry(key)
                .or_default()
                .insert(enc.into_iter().flatten().collect());
            if !reversed {
                fwd_key = Some(key);
                if rules::verify_rewrite(&gates, &[]) {
                    classes.entry(key).or_default().insert(Vec::new());
                }
            }
        }
    }
    classes
        .into_iter()
        .map(|(key, friends)| {
            let mut value = Vec::new();
            for f in friends {
                value.push(f.len() as u8);
                value.extend(f);
            }
            (key, value)
        })
        .collect()
}

fn g57_fixture_map() -> std::sync::Arc<std::collections::HashMap<[u8; 16], Vec<u8>>> {
    use std::sync::{Arc, OnceLock};
    static MAP: OnceLock<Arc<std::collections::HashMap<[u8; 16], Vec<u8>>>> = OnceLock::new();
    // Up to three gates on five wires: ~1.1k classes, built in about half a
    // second (release). Two-gate windows sampled convexly share a wire and
    // always fit; three-gate classes hold genuine re-spellings.
    MAP.get_or_init(|| Arc::new(g57_fixture_build(5, 3)))
        .clone()
}

fn g57_fixture_store() -> FrozenDb {
    FrozenDb::from_memory(g57_fixture_map())
}

// The stage-3 driver for tests: every round not taken by a twist, shuffle or
// bridge is a DB move against the fixture store, descending from four-gate
// windows, with the layer-2 MIX/COMP coin at p_mix = 0.3 so growth (MIX pays
// for longer spellings) and compression roughly balance: a 300-gate g57
// circuit ends near 470 gates after 20k moves with heavy lineage mixing.
fn db_walk_params() -> MixParams {
    MixParams {
        p_db: 1.0,
        s_db: 4,
        db_prefixes: true,
        p_mix: 0.3,
        ..MixParams::default()
    }
}
