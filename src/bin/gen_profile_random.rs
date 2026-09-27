//! Generate a STRUCTURELESS random circuit whose gate-type profile matches a
//! reference circuit — the control for refresh-length (L) measurements.
//!
//! Wires are chosen uniformly at random, so the circuit has no construction
//! structure at all; only the control-width histogram and comp fraction are
//! matched, so any difference in L reflects structure, not gate mix.
//!
//!   gen_profile_random <out.mpmct1> <wires> <gates> <seed> [--like <ref.mpmct1>]
//!
//! Without --like, uses the measured GSS-final profile (widths 0..8).
use local_mixing::postmix::format::{read_mpmct, write_mpmct};
use local_mixing::postmix::xgate::{Lits, XGate};

fn main() {
    let a: Vec<String> = std::env::args().collect();
    let out = &a[1];
    let wires: usize = a[2].parse().unwrap();
    let gates: usize = a[3].parse().unwrap();
    let mut st: u64 = a[4].parse().unwrap();

    // default profile: measured on circuits/gssmix128_indep_20260806/e1.5_h10_final
    let mut wcum: Vec<(usize, f64)> =
        vec![(0, 0.004), (1, 0.157), (2, 0.262), (3, 0.256), (4, 0.189), (5, 0.084), (6, 0.032), (7, 0.011), (8, 0.003)];
    let mut comp_frac = 0.012f64;

    // optional: match the reference's PER-WIRE usage frequency too (the real
    // circuits are ~10x overdispersed vs uniform, so a uniform control is not
    // actually "the same makeup of gates")
    let match_usage = a.iter().any(|s| s == "--match-wire-usage");
    let mut tgt_cum: Vec<f64> = Vec::new();
    let mut ctl_cum: Vec<f64> = Vec::new();

    if let Some(i) = a.iter().position(|s| s == "--like") {
        let (g, _) = read_mpmct(&a[i + 1]).expect("read reference");
        let mut hist = vec![0usize; 16];
        let mut ncomp = 0usize;
        if match_usage {
            let mut tw = vec![0f64; wires];
            let mut cw = vec![0f64; wires];
            for x in &g {
                tw[x.target as usize] += 1.0;
                for &(w, _) in &x.ctrls {
                    cw[w as usize] += 1.0;
                }
            }
            let mut acc = 0.0;
            for v in &tw {
                acc += *v;
                tgt_cum.push(acc);
            }
            let mut acc2 = 0.0;
            for v in &cw {
                acc2 += *v;
                ctl_cum.push(acc2);
            }
            eprintln!("[profile] matching per-wire usage (target total {acc}, control total {acc2})");
        }
        for x in &g {
            hist[x.ctrls.len().min(15)] += 1;
            ncomp += x.comp as usize;
        }
        let n = g.len() as f64;
        comp_frac = ncomp as f64 / n;
        wcum = hist
            .iter()
            .enumerate()
            .filter(|(_, c)| **c > 0)
            .map(|(w, c)| (w, *c as f64 / n))
            .collect();
        eprintln!("[profile] from {}: comp={:.4}, widths={:?}", a[i + 1], comp_frac,
            wcum.iter().map(|(w, p)| (*w, (p * 1000.0).round() / 1000.0)).collect::<Vec<_>>());
    }
    // to cumulative
    let mut acc = 0.0;
    for e in wcum.iter_mut() {
        acc += e.1;
        e.1 = acc;
    }
    let total = acc;

    let mut rnd = move || -> u64 {
        st = st.wrapping_add(0x9E3779B97F4A7C15);
        let mut z = st;
        z = (z ^ (z >> 30)).wrapping_mul(0xBF58476D1CE4E5B9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94D049BB133111EB);
        z ^ (z >> 31)
    };
    let mut unit = |r: &mut dyn FnMut() -> u64| -> f64 { (r() >> 11) as f64 / (1u64 << 53) as f64 };

    let mut out_gates: Vec<XGate> = Vec::with_capacity(gates);
    let mut used = vec![false; wires];
    while out_gates.len() < gates {
        let u = unit(&mut rnd) * total;
        let k = wcum.iter().find(|(_, c)| u <= *c).map(|(w, _)| *w).unwrap_or(2);
        let k = k.min(wires - 1);
        // distinct target + k controls; either uniform, or drawn from the
        // reference's empirical per-wire usage frequencies
        let draw = |cum: &Vec<f64>, r: &mut dyn FnMut() -> u64, wires: usize| -> u16 {
            if cum.is_empty() {
                return (r() % wires as u64) as u16;
            }
            let total = *cum.last().unwrap();
            let u = ((r() >> 11) as f64 / (1u64 << 53) as f64) * total;
            match cum.binary_search_by(|p| p.partial_cmp(&u).unwrap()) {
                Ok(i) => i as u16,
                Err(i) => i.min(wires - 1) as u16,
            }
        };
        let target = draw(&tgt_cum, &mut rnd, wires);
        used[target as usize] = true;
        let mut ctrls: Lits = Lits::new();
        let mut picked: Vec<u16> = Vec::with_capacity(k);
        let mut guard = 0;
        while picked.len() < k && guard < 1000 {
            let w = draw(&ctl_cum, &mut rnd, wires);
            if w != target && !used[w as usize] {
                used[w as usize] = true;
                picked.push(w);
            }
            guard += 1;
        }
        for w in &picked {
            ctrls.push((*w, rnd() & 1 == 1));
        }
        ctrls.sort_unstable();
        used[target as usize] = false;
        for w in &picked {
            used[*w as usize] = false;
        }
        let comp = unit(&mut rnd) < comp_frac;
        out_gates.push(XGate { target, comp, ctrls });
    }
    write_mpmct(out, &out_gates, wires).expect("write");
    println!("[genprof] wrote {} gates / {} wires to {}", out_gates.len(), wires, out);
}
