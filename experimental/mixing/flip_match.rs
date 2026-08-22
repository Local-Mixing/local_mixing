//! Endpoint-flip matching: does the gadget's carrier flip equal the source
//! circuit's gate flip?
//!
//! Tests the relation
//!     carrier(post update) XOR carrier(pre update)
//!       XOR  src_wire(post gate) XOR src_wire(pre gate)  =  0   (or 1: ledger const)
//!
//! i.e. the gadget moves a carrier by exactly the logical flip the source gate
//! applies. The single-carrier docs predict this is DEGREE ONE ("those masks
//! cancel, leaving carrier_before XOR carrier_after"), which the five/six/seven
//! carrier presets exist to break — this build does not use them.
//!
//! Method: both circuits run on the same x. For each source gate g on wire a,
//! delta_g = value_after XOR value_before (a function of x). In the predictor we
//! collect every wire segment's value; a matching pair (pre,post) on predictor
//! wire p exists iff  seg XOR delta_g  is also a segment value on wire p. That is
//! an O(1) hash lookup per candidate, so the search is exhaustive over pairs
//! without enumerating them.
//!
//! The pair is sought LOCALLY (within --kwin consecutive writes on a wire) and
//! exhaustively across ALL predictor wires: a gate-evaluation gadget updates its
//! carrier over a short run, so a whole-circuit-span pair is a boundary artifact,
//! not a per-gadget relation.
//!
//!   flip_match --source <s.mpmct1> --pred <p.mpmct1> [--samples N] [--kwin K]
use local_mixing::engine::format::read_mpmct;
use local_mixing::circuit::xgate::XGate;
use std::collections::HashMap;

fn seed(nw: usize, xs: &[u128], blk: usize) -> Vec<u64> {
    let mut st = vec![0u64; nw];
    for (l, &x) in xs.iter().enumerate() {
        for i in 0..blk {
            if (x >> i) & 1 == 1 {
                st[i] |= 1u64 << l;
            }
        }
    }
    st
}

fn classify(g: &XGate, n: usize, nf: usize) -> &'static str {
    let t = g.target as usize;
    if t >= n {
        if !g.comp && g.ctrls.len() == 1 && g.ctrls[0] == ((t - n) as u16, true) {
            return "N";
        }
        return "other-hi";
    }
    let hi = g.ctrls.iter().any(|&(w, _)| w as usize >= n);
    if hi {
        if nf == 0 { "S1" } else { "S2" }
    } else if nf == 0 { "C" } else if nf >= n { "D" } else { "C/D-in-N-band" }
}

fn main() {
    let a: Vec<String> = std::env::args().collect();
    let get = |k: &str| a.iter().position(|s| s == k).map(|i| a[i + 1].clone());
    let source = get("--source").expect("--source");
    let pred = get("--pred").expect("--pred");
    let samples: usize = get("--samples").map(|s| s.parse().unwrap()).unwrap_or(1024);
    let blk: usize = get("--blk").map(|s| s.parse().unwrap()).unwrap_or(128);
    assert!(samples % 64 == 0);
    let sw = samples / 64;

    let (sg, s_wires) = read_mpmct(&source).expect("source");
    let (pg, p_wires) = read_mpmct(&pred).expect("pred");
    eprintln!("source {} gates/{} wires ; pred {} gates/{} wires ; {} samples",
        sg.len(), s_wires, pg.len(), p_wires, samples);

    // ---- signatures: source per-gate flips, predictor per-segment values ----
    // src_flip[g] = value(target after g) XOR value(target before g)
    let mut src_flip = vec![0u64; sg.len() * sw];
    // predictor: for each wire, the list of its segment values (in order)
    let mut pred_seg: Vec<Vec<Vec<u64>>> = vec![Vec::new(); p_wires];
    let mut pred_pos: Vec<Vec<usize>> = vec![Vec::new(); p_wires];

    let mut rs = 0x243F6A8885A308D3u64;
    let mut rx = move || -> u128 {
        let mut sm = || -> u64 {
            rs = rs.wrapping_add(0x9E3779B97F4A7C15);
            let mut z = rs;
            z = (z ^ (z >> 30)).wrapping_mul(0xBF58476D1CE4E5B9);
            z = (z ^ (z >> 27)).wrapping_mul(0x94D049BB133111EB);
            z ^ (z >> 31)
        };
        let (h, l) = (sm(), sm());
        ((h as u128) << 64) | l as u128
    };
    let bmask: u128 = if blk >= 128 { u128::MAX } else { (1u128 << blk) - 1 };

    // predictor segment lists sized on first pass
    for w in 0..p_wires {
        pred_seg[w].push(vec![0u64; sw]); // initial segment
        pred_pos[w].push(0);
    }
    for g in pg.iter() {
        let t = g.target as usize;
        pred_seg[t].push(vec![0u64; sw]);
        pred_pos[t].push(0);
    }
    {
        let mut idx = vec![1usize; p_wires];
        for (gi, g) in pg.iter().enumerate() {
            let t = g.target as usize;
            pred_pos[t][idx[t]] = gi + 1;
            idx[t] += 1;
        }
    }

    for wi in 0..sw {
        let xs: Vec<u128> = (0..64).map(|_| rx() & bmask).collect();
        // source flips
        let mut st = seed(s_wires, &xs, blk);
        for (gi, g) in sg.iter().enumerate() {
            let t = g.target as usize;
            let before = st[t];
            g.apply_lanes(&mut st);
            src_flip[gi * sw + wi] = before ^ st[t];
        }
        // predictor segment values
        let mut st = seed(p_wires, &xs, blk);
        let mut idx = vec![0usize; p_wires];
        for w in 0..p_wires {
            pred_seg[w][0][wi] = st[w];
            idx[w] = 1;
        }
        for g in pg.iter() {
            let t = g.target as usize;
            g.apply_lanes(&mut st);
            pred_seg[t][idx[t]][wi] = st[t];
            idx[t] += 1;
        }
    }
    eprintln!("signatures done");

    // ---- exhaustive LOCAL-pair index over ALL predictor wires ----
    // A gate-evaluation gadget updates a carrier over a short run of writes, so
    // the (pre,post) pair we want is within a few segments on the same wire.
    // Index every such pair by a 64-bit hash of its difference, then look up each
    // source flip. Large-span pairs (whole-circuit brackets) are boundary
    // artifacts and are excluded by construction.
    let kwin: usize = get("--kwin").map(|s| s.parse().unwrap()).unwrap_or(12);
    let hash64 = |v: &[u64]| -> u64 {
        let mut h = 0xcbf29ce484222325u64;
        for &w in v { h ^= w; h = h.wrapping_mul(0x100000001b3); }
        h
    };
    let mut index: HashMap<u64, Vec<(u32, u32, u32)>> = HashMap::new();
    let mut npairs = 0usize;
    for w in 0..p_wires {
        let segs = &pred_seg[w];
        for i in 0..segs.len() {
            for j in (i + 1)..(i + 1 + kwin).min(segs.len()) {
                let d: Vec<u64> = segs[i].iter().zip(&segs[j]).map(|(a, b)| a ^ b).collect();
                if d.iter().all(|&x| x == 0) { continue; }
                index.entry(hash64(&d)).or_default().push((w as u32, i as u32, j as u32));
                npairs += 1;
            }
        }
    }
    eprintln!("local-pair index: {npairs} pairs (window {kwin} writes) over {p_wires} wires");

    let n_half = s_wires / 2;
    let mut nf = 0usize;
    let mut region = vec![""; sg.len()];
    for (gi, g) in sg.iter().enumerate() {
        let r = classify(g, n_half, nf);
        if r == "N" { nf += 1; }
        region[gi] = r;
    }

    let mut hits = 0usize;
    let mut hits_same_wire = 0usize;
    let mut trivial = 0usize;
    let mut spans: Vec<usize> = Vec::new();
    // where in the predictor does the matched pair sit, and on which wire block?
    // (a pair parked at the very end on a payload wire would be an assembly
    // artifact, not a distributed leak)
    let mut loc_frac: Vec<f64> = Vec::new();
    let mut loc_block: HashMap<&str, usize> = HashMap::new();
    let mut loc_frac_lin: Vec<f64> = Vec::new();
    let mut by_region: HashMap<&str, (usize, usize)> = HashMap::new();
    // gate TYPE: linear (CNOT / X) vs nonlinear (g57, CCNOT, wider conjunctions)
    let gtype = |g: &XGate| -> &'static str {
        if g.ctrls.len() <= 1 { "linear (CNOT/X)" } else { "nonlinear (>=2 ctrl)" }
    };
    let mut by_type: HashMap<&str, (usize, usize)> = HashMap::new();
    for gi in 0..sg.len() {
        let a = sg[gi].target as usize;
        let d = &src_flip[gi * sw..(gi + 1) * sw];
        let e = by_region.entry(region[gi]).or_insert((0, 0));
        e.1 += 1;
        let te = by_type.entry(gtype(&sg[gi])).or_insert((0, 0));
        if d.iter().all(|&x| x == 0) { trivial += 1; continue; }
        te.1 += 1;
        let mut found: Option<(usize, usize, usize)> = None;
        if let Some(cands) = index.get(&hash64(d)) {
            for &(w, i, j) in cands {
                // verify exactly (guard against the 64-bit hash colliding)
                let (w, i, j) = (w as usize, i as usize, j as usize);
                let ok = pred_seg[w][i].iter().zip(&pred_seg[w][j]).zip(d)
                    .all(|((x, y), z)| (x ^ y) == *z);
                if ok { found = Some((w, pred_pos[w][j].abs_diff(pred_pos[w][i]), pred_pos[w][i])); break; }
            }
        }
        if let Some((w, sp, wpos)) = found {
            hits += 1; e.0 += 1; by_type.get_mut(gtype(&sg[gi])).unwrap().0 += 1; spans.push(sp);
            if std::env::var("FLIP_MATCH_DUMP").is_ok() {
                eprintln!(
                    "[dump] src_gate={gi} region={} type={} src_target={a} pred_wire={w} \
                     pair_gate_pos={wpos}..{} span_gates={sp}",
                    region[gi], gtype(&sg[gi]), wpos + sp
                );
            }
            let fr = wpos as f64 / pg.len() as f64;
            loc_frac.push(fr);
            if gtype(&sg[gi]) == "linear (CNOT/X)" { loc_frac_lin.push(fr); }
            *loc_block.entry(if w < 128 { "data0(0-127)" } else if w < 256 { "payload(128-255)" }
                             else if w < 384 { "bandA(256-383)" } else { "bandB(384-511)" }).or_insert(0) += 1;
            if w == a { hits_same_wire += 1; }
        }
    }

    println!("=== LOCAL endpoint-flip match: source={} pred={} ===", source, pred);
    println!("source gates                      : {}", sg.len());
    println!("  flips identically 0 (excluded)  : {trivial}");
    println!("  local carrier pairs indexed     : {npairs}");
    println!("\nmatches (carrier_post ^ carrier_pre == source gate flip):");
    println!("  gates matched                   : {hits} / {} ({:.3}%)",
        sg.len() - trivial, 100.0 * hits as f64 / (sg.len() - trivial) as f64);
    println!("  ...on the same wire index       : {hits_same_wire}");
    if !spans.is_empty() {
        let mut v = spans.clone(); v.sort_unstable();
        println!("  gate-span of matched pair       : min {} median {} max {}", v[0], v[v.len()/2], v[v.len()-1]);
    }
    if !loc_frac.is_empty() {
        let mut v = loc_frac.clone(); v.sort_by(|a,b| a.partial_cmp(b).unwrap());
        println!("  matched-pair POSITION in predictor (fraction of depth):");
        println!("     min {:.3}  p25 {:.3}  median {:.3}  p75 {:.3}  max {:.3}",
            v[0], v[v.len()/4], v[v.len()/2], v[3*v.len()/4], v[v.len()-1]);
        let late = v.iter().filter(|&&f| f > 0.9).count();
        println!("     in the last 10% of the circuit: {} / {} ({:.1}%)", late, v.len(), 100.0*late as f64/v.len() as f64);
        if !loc_frac_lin.is_empty() {
            let mut l = loc_frac_lin.clone(); l.sort_by(|a,b| a.partial_cmp(b).unwrap());
            let latel = l.iter().filter(|&&f| f > 0.9).count();
            println!("     LINEAR-gate matches: median {:.3}, last 10%: {} / {} ({:.1}%)",
                l[l.len()/2], latel, l.len(), 100.0*latel as f64/l.len() as f64);
        }
        println!("  matched-pair predictor wire block:");
        let mut bk: Vec<&&str> = loc_block.keys().collect(); bk.sort();
        for k in bk { println!("     {:<18} {}", k, loc_block[*k]); }
    }
    println!("  by GATE TYPE (matched / total, zero-flip gates excluded):");
    let mut tk: Vec<&&str> = by_type.keys().collect();
    tk.sort();
    for k in tk {
        let (h, t) = by_type[*k];
        if t > 0 { println!("    {:<22} {:>6} / {:<6} {:>7.2}%", k, h, t, 100.0 * h as f64 / t as f64); }
    }
    println!("  by region (matched / total):");
    let mut ks: Vec<&&str> = by_region.keys().collect();
    ks.sort();
    for k in ks {
        let (h, t) = by_region[*k];
        println!("    {:<16} {:>6} / {:<6} {:>7.2}%", k, h, t, 100.0 * h as f64 / t as f64);
    }
}
