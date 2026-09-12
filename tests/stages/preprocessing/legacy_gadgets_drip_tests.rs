use super::*;
use rand::{Rng, SeedableRng, rngs::StdRng};

fn apply(gates: &[XGate], state: &mut [bool]) {
    for g in gates {
        let conj = g.ctrls.iter().all(|&(w, pol)| state[w as usize] == pol);
        if g.comp ^ conj {
            state[g.target as usize] ^= true;
        }
    }
}

fn random_source(n: usize, m: usize, rng: &mut impl Rng) -> Vec<XGate> {
    (0..m)
        .map(|_| {
            let target = rng.random_range(0..n) as u16;
            let width = rng.random_range(0..3usize); // 0, 1, or 2 controls
            let mut ctrls: Vec<(u16, bool)> = Vec::new();
            let mut used = vec![target];
            for _ in 0..width {
                loop {
                    let w = rng.random_range(0..n) as u16;
                    if !used.contains(&w) {
                        used.push(w);
                        ctrls.push((w, rng.random_range(0..2) == 1));
                        break;
                    }
                }
            }
            let comp = rng.random_range(0..2) == 1;
            XGate::conj(target, ctrls.iter().copied())
                .map(|mut g| {
                    g.comp = comp;
                    g
                })
                .unwrap_or_else(|| XGate::x_gate(target))
        })
        .collect()
}

#[test]
fn drip_gadget_computes_c_and_reports_size() {
    for &k in &[1usize, 2] {
        let mut rng = StdRng::seed_from_u64(0xd21b_9c4e ^ k as u64);
        let n = 24;
        let m = 60;
        let source = random_source(n, m, &mut rng);
        let gadget = gadgetize_drip_single(&source, n, k, &mut rng);
        let total = gadget.num_wires;
        for _ in 0..64 {
            let x: Vec<bool> = (0..n).map(|_| rng.random_range(0..2) == 1).collect();
            let mut refs = vec![false; n];
            refs.copy_from_slice(&x);
            apply(&source, &mut refs);
            let mut gs = vec![false; total];
            gs[..n].copy_from_slice(&x);
            apply(&gadget.gates, &mut gs);
            assert_eq!(&gs[..n], &refs[..], "drip k={k} output mismatch");
        }
        eprintln!(
            "[drip] k={k} n={n} |source|={m} -> gadget={} gates ({:.1}x/gate), wires={total}",
            gadget.gates.len(),
            gadget.gates.len() as f64 / m as f64
        );
    }

    // Regional (mixed depth: k=1 baseline, k=2 on a middle window) must
    // also compute C exactly -- each gate raises/fires/lowers on its own.
    let mut rng = StdRng::seed_from_u64(0xbeef_1234);
    let n = 24;
    let m = 60;
    let source = random_source(n, m, &mut rng);
    let gadget = gadgetize_drip_regional(&source, n, 1, 2, 20..40, &mut rng);
    let total = gadget.num_wires;
    for _ in 0..64 {
        let x: Vec<bool> = (0..n).map(|_| rng.random_range(0..2) == 1).collect();
        let mut refs = vec![false; n];
        refs.copy_from_slice(&x);
        apply(&source, &mut refs);
        let mut gs = vec![false; total];
        gs[..n].copy_from_slice(&x);
        apply(&gadget.gates, &mut gs);
        assert_eq!(&gs[..n], &refs[..], "regional drip output mismatch");
    }
}

#[test]
fn drip_layered_computes_c_and_reports_size() {
    for &nr in &[0usize, 4] {
        let mut rng = StdRng::seed_from_u64(0x1a4e_2200 ^ nr as u64);
        let n = 32;
        let m = 150;
        let source = random_source(n, m, &mut rng);
        let gadget = gadgetize_drip_layered(&source, n, nr, &mut rng);
        let total = gadget.num_wires;
        let mut g57 = 0usize;
        for g in &gadget.gates {
            if g.comp && g.ctrls.len() == 2 {
                g57 += 1;
            }
        }
        for _ in 0..64 {
            let x: Vec<bool> = (0..n).map(|_| rng.random_range(0..2) == 1).collect();
            let mut refs = vec![false; n];
            refs.copy_from_slice(&x);
            apply(&source, &mut refs);
            let mut gs = vec![false; total];
            gs[..n].copy_from_slice(&x);
            apply(&gadget.gates, &mut gs);
            assert_eq!(&gs[..n], &refs[..], "layered drip n_rerand={nr} mismatch");
        }
        let maxw = gadget
            .gates
            .iter()
            .map(|g| g.ctrls.len())
            .max()
            .unwrap_or(0);
        eprintln!(
            "[drip-layered] n={n} |src|={m} n_rerand={nr} -> {} gates ({:.1}x), g57={:.0}%, maxw={maxw}",
            gadget.gates.len(),
            gadget.gates.len() as f64 / m as f64,
            100.0 * g57 as f64 / gadget.gates.len() as f64
        );
    }
}

// Two-layer seam prototype: molecules (self-contained 2-wire sub-circuits)
// routed among isomorphic siblings, fired position-agnostically, with the
// permutation kept LIVE across the seam (no route-back). Compares against a
// route-back variant. Exhaustive over all data AND all routing-aux settings.
#[test]
fn seam_molecule_prototype() {
    const NMOL: usize = 4;
    const NDATA: usize = 2 * NMOL; // 8
    const ZERO: usize = NDATA; // wire 8 = constant 0 (never set)
    const ABASE: usize = NDATA + 1; // routing bits: wires 9..15
    const NBITS: usize = 6;
    const NTOT: usize = ABASE + NBITS; // 15
    const NVAR: usize = NDATA + NBITS; // 14 varying input bits
    const NPT: usize = 1 << NVAR; // 16384
    const WORDS: usize = NPT / 64;

    // Clean aux-controlled swap of wires p,q iff bit z=1 (CSWITCH4, z2=ZERO).
    fn cswap(p: usize, q: usize, z: usize, out: &mut Vec<XGate>) {
        out.push(XGate::cnot(p as u16, q as u16));
        out.push(XGate::from_g57([q as u16, z as u16, p as u16]));
        out.push(XGate::from_g57([q as u16, ZERO as u16, p as u16]));
        out.push(XGate::cnot(p as u16, q as u16));
    }
    // Swap whole molecules i,j (both rails share the same control bit z).
    fn mol_swap(i: usize, j: usize, z: usize, out: &mut Vec<XGate>) {
        cswap(2 * i, 2 * j, z, out);
        cswap(2 * i + 1, 2 * j + 1, z, out);
    }
    // A connected route stage over 4 molecules using 3 control bits.
    fn route(bits: [usize; 3], out: &mut Vec<XGate>) {
        let swaps = [(0usize, 1usize, bits[0]), (2, 3, bits[1]), (1, 2, bits[2])];
        for &(i, j, z) in &swaps {
            mol_swap(i, j, z, out);
        }
    }
    fn unroute(bits: [usize; 3], out: &mut Vec<XGate>) {
        let swaps = [(1usize, 2usize, bits[2]), (2, 3, bits[1]), (0, 1, bits[0])];
        for &(i, j, z) in &swaps {
            mol_swap(i, j, z, out);
        }
    }
    fn layer1(out: &mut Vec<XGate>) {
        for i in 0..NMOL {
            out.push(XGate::cnot((2 * i) as u16, (2 * i + 1) as u16)); // a ^= b
        }
    }
    fn layer2(out: &mut Vec<XGate>) {
        for i in 0..NMOL {
            out.push(XGate::cnot((2 * i + 1) as u16, (2 * i) as u16)); // b ^= a
        }
    }

    let a = [9usize, 10, 11];
    let b = [12usize, 13, 14];

    // Ideal C (no aux): layer1 then layer2.
    let mut ideal = Vec::new();
    layer1(&mut ideal);
    let seam_ideal = ideal.len();
    layer2(&mut ideal);

    // PERSIST: route A; fire L1; [seam]; route B; fire L2; then un-route to
    // canonical output. Permutation stays live across the seam.
    let mut persist = Vec::new();
    route(a, &mut persist);
    layer1(&mut persist);
    let seam_persist = persist.len();
    route(b, &mut persist);
    layer2(&mut persist);
    unroute(b, &mut persist);
    unroute(a, &mut persist);

    // ROUTEBACK: route A; fire L1; un-route A; [seam canonical]; route B; L2; un-route B.
    let mut routeback = Vec::new();
    route(a, &mut routeback);
    layer1(&mut routeback);
    unroute(a, &mut routeback);
    let seam_rb = routeback.len();
    route(b, &mut routeback);
    layer2(&mut routeback);
    unroute(b, &mut routeback);

    // ---- exhaustive correctness: for every (data, routing bits), both
    // gadgets must compute ideal C on the 8 data wires (wire8 pinned 0) ----
    let set_state = |asg: usize| -> Vec<bool> {
        let mut s = vec![false; NTOT];
        for w in 0..NDATA {
            s[w] = (asg >> w) & 1 == 1;
        }
        for k in 0..NBITS {
            s[ABASE + k] = (asg >> (NDATA + k)) & 1 == 1;
        }
        s // wire ZERO stays false
    };
    for asg in 0..NPT {
        let base = set_state(asg);
        let mut want = base.clone();
        apply(&ideal, &mut want);
        for (name, g) in [("persist", &persist), ("routeback", &routeback)] {
            let mut got = base.clone();
            apply(g, &mut got);
            assert_eq!(&got[..NDATA], &want[..NDATA], "{name} wrong C at asg {asg}");
        }
    }

    // ---- exposure at the seam: how many of ideal's layer-1 output
    // functions lie in the GF(2) affine span of the gadget's data-wire
    // functions at its seam cut (exhaustive over all NPT inputs) ----
    let func_at = |gates: &[XGate], cut: usize, wire: usize| -> Vec<u64> {
        let mut v = vec![0u64; WORDS];
        for asg in 0..NPT {
            let mut s = set_state(asg);
            for g in &gates[..cut] {
                let conj = g.ctrls.iter().all(|&(w, p)| s[w as usize] == p);
                if g.comp ^ conj {
                    s[g.target as usize] ^= true;
                }
            }
            if s[wire] {
                v[asg >> 6] |= 1u64 << (asg & 63);
            }
        }
        v
    };
    let xor = |a: &mut Vec<u64>, b: &[u64]| {
        for i in 0..WORDS {
            a[i] ^= b[i];
        }
    };
    // Build a reduced basis of the predictor span (+constant 1), return #targets in span.
    let count_in_span = |preds: &[Vec<u64>], targets: &[Vec<u64>]| -> usize {
        let mut ones = vec![0u64; WORDS];
        for w in ones.iter_mut() {
            *w = !0u64;
        }
        let mut basis: Vec<Vec<u64>> = vec![ones];
        for p in preds {
            let mut r = p.clone();
            for bvec in &basis {
                let piv = bvec.iter().position(|&x| x != 0);
                if let Some(pi) = piv {
                    let bit = bvec[pi].trailing_zeros();
                    if (r[pi] >> bit) & 1 == 1 {
                        xor(&mut r, bvec);
                    }
                }
            }
            if r.iter().any(|&x| x != 0) {
                basis.push(r);
            }
        }
        let reduce = |mut r: Vec<u64>| -> bool {
            for bvec in &basis {
                let pi = bvec.iter().position(|&x| x != 0).unwrap();
                let bit = bvec[pi].trailing_zeros();
                if (r[pi] >> bit) & 1 == 1 {
                    xor(&mut r, bvec);
                }
            }
            r.iter().all(|&x| x == 0)
        };
        targets.iter().filter(|t| reduce((*t).clone())).count()
    };

    let targets: Vec<Vec<u64>> = (0..NDATA).map(|w| func_at(&ideal, seam_ideal, w)).collect();
    let preds_p: Vec<Vec<u64>> = (0..NDATA)
        .map(|w| func_at(&persist, seam_persist, w))
        .collect();
    let preds_r: Vec<Vec<u64>> = (0..NDATA)
        .map(|w| func_at(&routeback, seam_rb, w))
        .collect();

    let exp_p = count_in_span(&preds_p, &targets);
    let exp_r = count_in_span(&preds_r, &targets);
    eprintln!(
        "[seam] correctness OK (all {NPT} inputs x aux). Layer-1 segments affinely exposed at seam:  PERSIST {exp_p}/{NDATA}   ROUTEBACK {exp_r}/{NDATA}"
    );
    assert!(
        exp_p < exp_r,
        "persist should expose fewer seam segments than route-back"
    );
}

#[test]
fn drip_layered_on_sandwich_source_computes_c() {
    use crate::circuit::random_circuit;
    for &nr in &[0usize, 4] {
        let mut rng = StdRng::seed_from_u64(0x5a2d_0000 ^ nr as u64);
        let n_c = 16;
        let m = 100;
        let c = random_circuit(n_c, m);
        let s = sandwich_default_s(n_c);
        let sandwich = sliced_sandwich_cnot(&c, n_c, m, s, SandwichVariant::Classic, &mut rng);
        let sn = sandwich.num_wires;
        let g = gadgetize_drip_layered(&sandwich.gates, sn, nr, &mut rng);
        let total = g.num_wires;
        for _ in 0..48 {
            let x: Vec<bool> = (0..sn).map(|_| rng.random_range(0..2) == 1).collect();
            let mut refs = vec![false; sn];
            refs.copy_from_slice(&x);
            apply(&sandwich.gates, &mut refs);
            let mut gs = vec![false; total];
            gs[..sn].copy_from_slice(&x);
            apply(&g.gates, &mut gs);
            assert_eq!(
                &gs[..sn],
                &refs[..],
                "drip on SANDWICH source mismatch nr={nr}"
            );
        }
    }
}

#[test]
fn drip_persist_on_sandwich_source_computes_c() {
    use crate::circuit::random_circuit;
    let mut rng = StdRng::seed_from_u64(0x9e5a_11cc);
    let n_c = 16;
    let m = 100;
    let c = random_circuit(n_c, m);
    let s = sandwich_default_s(n_c);
    let sandwich = sliced_sandwich_cnot(&c, n_c, m, s, SandwichVariant::Classic, &mut rng);
    let sn = sandwich.num_wires;
    let g = gadgetize_drip_persist(&sandwich.gates, sn, &mut rng);
    let total = g.num_wires;
    for _ in 0..48 {
        let x: Vec<bool> = (0..sn).map(|_| rng.random_range(0..2) == 1).collect();
        let mut refs = vec![false; sn];
        refs.copy_from_slice(&x);
        apply(&sandwich.gates, &mut refs);
        let mut gs = vec![false; total];
        gs[..sn].copy_from_slice(&x);
        apply(&g.gates, &mut gs);
        assert_eq!(&gs[..sn], &refs[..], "persist on SANDWICH source mismatch");
    }
}

#[test]
fn wrap_drip_computes_c_on_upper_half() {
    use crate::circuit::random_circuit;
    let mut rng = StdRng::seed_from_u64(0x11a2_5e6d);
    let n_c = 16;
    let m = 120;
    let c = random_circuit(n_c, m);
    let s = sandwich_default_s(n_c);
    let sandwich = sliced_sandwich_cnot(&c, n_c, m, s, SandwichVariant::Classic, &mut rng);
    let sn = sandwich.num_wires;
    let wrapped = wrap_drip_delivery(&sandwich.gates, sn, 4, 4 * sn, &mut rng);
    let total = wrapped.num_wires;
    let half = sn / 2;
    for _ in 0..48 {
        let x: Vec<bool> = (0..n_c).map(|_| rng.random_range(0..2) == 1).collect();
        let mut sin = vec![false; sn];
        sin[..n_c].copy_from_slice(&x);
        apply(&sandwich.gates, &mut sin);
        let mut ws = vec![false; total];
        ws[..n_c].copy_from_slice(&x);
        apply(&wrapped.gates, &mut ws);
        assert_eq!(
            &ws[half..sn],
            &sin[half..sn],
            "wrap upper-half payload mismatch"
        );
    }
    eprintln!(
        "[wrap] n_c={n_c} sandwich={}g -> wrapped={}g/{total}w (upper-half payload verified)",
        sandwich.gates.len(),
        wrapped.gates.len()
    );
}

#[test]
#[ignore = "emits mpmct1 files for segment_deduce measurement"]
fn drip_emit_for_segment_deduce() {
    use crate::circuit::random_circuit;
    use crate::engine::format::write_mpmct;
    let dir = "/private/tmp/claude-501/-Users-rancanetti-Documents-local-mixing/659c6d96-f555-40bf-8d53-8d6a9dad9abf/scratchpad";
    let mut rng = StdRng::seed_from_u64(0x5e6d_2026);
    let n_c = 32;
    let m = 400;
    let c = random_circuit(n_c, m);
    let s = sandwich_default_s(n_c);
    let sandwich = sliced_sandwich_cnot(&c, n_c, m, s, SandwichVariant::Classic, &mut rng);
    let sn = sandwich.num_wires; // 64
    write_mpmct(&format!("{dir}/sd_sandwich.mpmct1"), &sandwich.gates, sn).unwrap();
    // Drip gadget for several aux-rerandomization densities.
    for &nr in &[0usize, 4, 16, 64] {
        let g = gadgetize_drip_layered(&sandwich.gates, sn, nr, &mut rng);
        write_mpmct(
            &format!("{dir}/sd_gadget_nr{nr}.mpmct1"),
            &g.gates,
            g.num_wires,
        )
        .unwrap();
        eprintln!(
            "[emit] n_rerand={nr}: gadget={}g/{}w",
            g.gates.len(),
            g.num_wires
        );
    }
    // Persist-mode gadget (no route-back).
    let gp = gadgetize_drip_persist(&sandwich.gates, sn, &mut rng);
    write_mpmct(&format!("{dir}/sd_persist.mpmct1"), &gp.gates, gp.num_wires).unwrap();
    eprintln!(
        "[emit] persist: gadget={}g/{}w",
        gp.gates.len(),
        gp.num_wires
    );
    // No-reshuffle positive control: sandwich fired in place + band fill on the 2n frame.
    let mut ctrl = Vec::new();
    let band: Vec<u16> = (sn..2 * sn).map(|w| w as u16).collect();
    emit_band_fill_src(sn, &band, &mut rng, &mut ctrl);
    ctrl.extend(sandwich.gates.iter().cloned());
    write_mpmct(&format!("{dir}/sd_control.mpmct1"), &ctrl, 2 * sn).unwrap();
    eprintln!(
        "[emit] n_c={n_c} blk={n_c} sandwich={}g/{sn}w control={}g -> {dir}/sd_*.mpmct1",
        sandwich.gates.len(),
        ctrl.len()
    );
}

#[test]
#[ignore = "heavy: layered drip on a real |C|=|D|=3000 sandwich"]
fn drip_layered_size_on_real_sandwich() {
    use crate::circuit::random_circuit;
    let mut rng = StdRng::seed_from_u64(0x1a4e_c0de);
    let n_c = 128;
    let m = 3000;
    let c = random_circuit(n_c, m);
    let s = sandwich_default_s(n_c);
    let sandwich = sliced_sandwich_cnot(&c, n_c, m, s, SandwichVariant::Classic, &mut rng);
    let sn = sandwich.num_wires;
    let raw = sandwich.gates.len();
    let half = sn / 2;
    let in_group: Vec<bool> = sandwich
        .gates
        .iter()
        .map(|g| is_n_column(g, half))
        .collect();
    let layers = layer_wire_disjoint_grouped(&sandwich.gates, 2 * sn, &in_group);
    let maxlayer = layers.iter().map(|l| l.len()).max().unwrap_or(0);
    let avglayer = raw as f64 / layers.len() as f64;
    // max type-class size within any layer (the routable-class size)
    let mut maxclass = 0usize;
    for l in &layers {
        let mut cnt: std::collections::HashMap<(bool, Vec<bool>), usize> =
            std::collections::HashMap::new();
        for &gi in l {
            *cnt.entry(class_signature(&sandwich.gates[gi])).or_default() += 1;
        }
        maxclass = maxclass.max(cnt.values().copied().max().unwrap_or(0));
    }
    eprintln!("[drip-layered] N-column gates grouped; max type-class in a layer = {maxclass}");
    for &nr in &[0usize, 8] {
        let g = gadgetize_drip_layered(&sandwich.gates, sn, nr, &mut rng);
        let (mut g57, mut wide, mut maxw) = (0usize, 0usize, 0usize);
        for gate in &g.gates {
            let w = gate.ctrls.len();
            maxw = maxw.max(w);
            if gate.comp && w == 2 {
                g57 += 1;
            }
            if w > 2 {
                wide += 1;
            }
        }
        eprintln!(
            "[drip-layered] raw={raw}/{sn}w, {} layers (max {maxlayer}, avg {avglayer:.1}) | n_rerand={nr}: {} gates ({:.0}x), g57={:.0}%, wide={wide}, maxw={maxw}, wires={}",
            layers.len(),
            g.gates.len(),
            g.gates.len() as f64 / raw as f64,
            100.0 * g57 as f64 / g.gates.len() as f64,
            g.num_wires
        );
    }
}

#[test]
#[ignore = "heavy: builds a real |C|=|D|=3000 sandwich and drips it"]
fn drip_size_on_real_sliced_sandwich() {
    use crate::circuit::random_circuit;
    let mut rng = StdRng::seed_from_u64(0x5117_ced5);
    let n_c = 128;
    let m = 3000;
    let c = random_circuit(n_c, m);
    let s = sandwich_default_s(n_c);
    let sandwich = sliced_sandwich_cnot(&c, n_c, m, s, SandwichVariant::Classic, &mut rng);
    let sn = sandwich.num_wires; // 256
    let raw = sandwich.gates.len();

    // Locate the N-column bridge gates: CNOT high<-low, i.e. y_i ^= x_i.
    let n_pos: Vec<usize> = sandwich
        .gates
        .iter()
        .enumerate()
        .filter(|(_, g)| {
            g.ctrls.len() == 1
                && !g.comp
                && (g.target as usize) >= n_c
                && (g.ctrls[0].0 as usize) < n_c
                && g.ctrls[0].1
        })
        .map(|(i, _)| i)
        .collect();
    let n_lo = n_pos.iter().copied().min();
    let n_hi = n_pos.iter().copied().max();
    eprintln!("[drip-size] raw sliced sandwich = {raw} gates / {sn} wires (|C|=|D|={m})");
    eprintln!(
        "[drip-size] N-form bridge gates: count={}, span={:?}..{:?}",
        n_pos.len(),
        n_lo,
        n_hi
    );

    // Sensitive window = N span padded by n_c each side (fallback: middle).
    let margin = n_c;
    let sensitive = match (n_lo, n_hi) {
        (Some(a), Some(b)) => a.saturating_sub(margin)..(b + margin + 1).min(raw),
        _ => (raw / 2).saturating_sub(2 * n_c)..(raw / 2 + 2 * n_c).min(raw),
    };
    let win = sensitive.end.saturating_sub(sensitive.start);

    // Report a build: pre/post-frag size and the gate-type histogram
    // (indexed by number of controls; 0=NOT, 1=CNOT, 2=Toffoli/g57-form,
    // >=3 = wide multiplex copies). Everything the drip emits is comp=0.
    let report = |label: &str, g: &CnotCircuit| {
        let (mut frag, mut maxw, mut comp) = (0usize, 0usize, 0usize);
        let mut hist = [0usize; 12];
        for gate in &g.gates {
            let w = gate.ctrls.len();
            maxw = maxw.max(w);
            hist[w.min(11)] += 1;
            if gate.comp {
                comp += 1;
            }
            frag += if w > 2 { 2 * (w - 2) + 1 } else { 1 };
        }
        let total = g.gates.len();
        eprintln!(
            "[drip-size] {label}: pre={total} ({:.0}x), post-frag~{frag} ({:.0}x), maxw={maxw}, comp=1 gates={comp}",
            total as f64 / raw as f64,
            frag as f64 / raw as f64
        );
        let pct = |c: usize| 100.0 * c as f64 / total as f64;
        eprintln!(
            "[drip-size]   makeup: NOT(0c)={} CNOT(1c)={} ({:.0}%) | 2c={} ({:.0}%) | 3c={} 4c={} 5c={} 6c={} 7c={} 8c={} | wide(>2c)={} ({:.0}%)",
            hist[0],
            hist[1],
            pct(hist[1]),
            hist[2],
            pct(hist[2]),
            hist[3],
            hist[4],
            hist[5],
            hist[6],
            hist[7],
            hist[8],
            hist[3..].iter().sum::<usize>(),
            pct(hist[3..].iter().sum())
        );
    };

    report(
        "BUILD A (k=1 all)          ",
        &gadgetize_drip_regional(&sandwich.gates, sn, 1, 1, 0..0, &mut rng),
    );
    report(
        &format!(
            "BUILD B (k=1,k=2 mid {:.0}%)  ",
            100.0 * win as f64 / raw as f64
        ),
        &gadgetize_drip_regional(&sandwich.gates, sn, 1, 2, sensitive.clone(), &mut rng),
    );
    report(
        "ref     (k=2 all)          ",
        &gadgetize_drip_regional(&sandwich.gates, sn, 2, 2, 0..0, &mut rng),
    );
}
