//! Per-width commutation-leeway census of a circuit file: for every gate, the
//! two-sided float-box size (steps past non-colliders in each direction until
//! the first collider, capped per direction — the exact `XGate::collides` /
//! fmix `float_distance` semantics), binned by control count and split by
//! comp (g57-class, no opposite-literal separation exemption) vs plain
//! conjunction (exempt: an opposite shared literal separates). This is the
//! static mobility curve D(k) — the direct test of "wider gates have more
//! leeway to move".
//!
//! Usage: leeway_by_width <circuit.mpmct1> [cap=4096] [format=mpmct1|g57]

use local_mixing::engine::format::{read_g57_file, read_mpmct};
use local_mixing::circuit::xgate::{XGate, max_wire};

fn pct(v: &[usize], p: f64) -> usize {
    if v.is_empty() {
        return 0;
    }
    let i = ((v.len() - 1) as f64 * p).round() as usize;
    v[i]
}

fn main() {
    let mut a = std::env::args().skip(1);
    let path = a.next().expect("usage: leeway_by_width <circuit> [cap] [format]");
    let cap: usize = a.next().and_then(|s| s.parse().ok()).unwrap_or(4096);
    let format = a.next().unwrap_or_else(|| "mpmct1".to_string());
    let gates: Vec<XGate> = match format.as_str() {
        "mpmct1" => read_mpmct(&path).expect("read mpmct1").0,
        "g57" => read_g57_file(&path).expect("read g57"),
        other => panic!("unknown format {other}"),
    };
    let n = gates.len();
    eprintln!(
        "[leeway] {} gates, {} wires, cap {}/side",
        n,
        max_wire(&gates) as usize + 1,
        cap
    );

    // Two-sided float box per gate.
    let leeway = |i: usize| -> usize {
        let g = &gates[i];
        let mut d = 0usize;
        let mut j = i;
        while j > 0 && d < cap && !XGate::collides(g, &gates[j - 1]) {
            j -= 1;
            d += 1;
        }
        let mut r = 0usize;
        let mut k = i + 1;
        while k < n && r < cap && !XGate::collides(g, &gates[k]) {
            k += 1;
            r += 1;
        }
        d + r
    };

    // width -> (plain leeways, comp leeways)
    let mut bins: Vec<(Vec<usize>, Vec<usize>)> = Vec::new();
    for i in 0..n {
        let w = gates[i].ctrls.len();
        if bins.len() <= w {
            bins.resize_with(w + 1, || (Vec::new(), Vec::new()));
        }
        let l = leeway(i);
        if gates[i].comp {
            bins[w].1.push(l);
        } else {
            bins[w].0.push(l);
        }
    }

    println!(
        "{:>5} {:>9} {:>9} {:>7} {:>7} | {:>9} {:>9} {:>7} {:>7}",
        "width", "n_plain", "mean_pl", "med_pl", "p90_pl", "n_comp", "mean_cp", "med_cp", "p90_cp"
    );
    for (w, (mut pl, mut cp)) in bins.into_iter().enumerate() {
        if pl.is_empty() && cp.is_empty() {
            continue;
        }
        pl.sort_unstable();
        cp.sort_unstable();
        let mean = |v: &[usize]| {
            if v.is_empty() {
                0.0
            } else {
                v.iter().sum::<usize>() as f64 / v.len() as f64
            }
        };
        println!(
            "{:>5} {:>9} {:>9.1} {:>7} {:>7} | {:>9} {:>9.1} {:>7} {:>7}",
            w,
            pl.len(),
            mean(&pl),
            pct(&pl, 0.5),
            pct(&pl, 0.9),
            cp.len(),
            mean(&cp),
            pct(&cp, 0.5),
            pct(&cp, 0.9)
        );
    }
}
