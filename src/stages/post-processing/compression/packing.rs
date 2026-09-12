//! Canonical ANF packing and deterministic compact ESOP spelling.
use super::*;

// ---- packing ---------------------------------------------------------------
//
// At a fixed point of the gather every maximal run of consecutive same-target
// gates is one gathered group (a group floats to one close point and is
// emitted there, and runs that could still float together would have been
// merged). Packing spells each run as ONE generalized gate t ^= f(controls)
// with f in algebraic normal form -- the XOR of positive monomials, the
// unique representation of a Boolean function. The point is not size (the
// ANF is ~2.4x the cube count on GSS finals) but the removal of information:
// the cube list fcompress emits is the catalogue-reduced descendant of the
// cubes the mixer left, so it carries history, while the ANF depends on
// nothing but the function. Exact and attacker-computable like the rest of
// the pass. Monomials are ascending wire lists sorted by (degree, wires);
// the empty monomial is the constant 1 (a comp bit). Any support size.
pub(super) fn pack_run(run: &[XGate]) -> PackedGate {
    let target = run[0].target;
    let mut set: FxHashSet<Vec<u16>> = FxHashSet::default();
    let mut toggle = |m: Vec<u16>| {
        if !set.remove(&m) {
            set.insert(m);
        }
    };
    for g in run {
        debug_assert_eq!(g.target, target);
        if g.comp {
            toggle(Vec::new());
        }
        let pos: Vec<u16> = g.ctrls.iter().filter(|l| l.1).map(|l| l.0).collect();
        let neg: Vec<u16> = g.ctrls.iter().filter(|l| !l.1).map(|l| l.0).collect();
        assert!(
            neg.len() <= 24,
            "cube with {} negative literals: expansion too large",
            neg.len()
        );
        // AND(pos) * PROD(1 XOR w in neg) = XOR over subsets of neg.
        for sub in 0..(1u64 << neg.len()) {
            let mut m = pos.clone();
            for (b, &w) in neg.iter().enumerate() {
                if sub >> b & 1 == 1 {
                    m.push(w);
                }
            }
            m.sort_unstable();
            toggle(m);
        }
    }
    let mut g = PackedGate {
        target,
        terms: set
            .into_iter()
            .map(|m| m.into_iter().map(|w| (w, true)).collect())
            .collect(),
    };
    g.sort_terms();
    g
}

/// Compaction: rewrite a packed ANF gate as a mixed-polarity ESOP by the
/// deterministic reducer strategies, from the ANF ALONE (never from the
/// cubes the ANF came from), so the result is still one spelling per
/// activation function -- at ~2.3x fewer terms than the ANF. Gates whose
/// support exceeds the 63-wire mask width are left in ANF (still canonical).
pub fn compact_gate(g: &PackedGate) -> PackedGate {
    debug_assert!(g.is_anf(), "compaction starts from the ANF");
    let mut support: Vec<u16> = g.terms.iter().flatten().map(|l| l.0).collect();
    support.sort_unstable();
    support.dedup();
    if support.len() > 63 {
        return g.clone();
    }
    let mut monos: Vec<u64> = g
        .terms
        .iter()
        .map(|t| {
            t.iter().fold(0u64, |m, &(w, _)| {
                m | 1u64 << support.binary_search(&w).expect("wire in support")
            })
        })
        .collect();
    monos.sort_unstable();
    let (cubes, parity, _) = esop_from_monomials(&support, &monos);
    let mut out = PackedGate {
        target: g.target,
        terms: cubes.into_iter().map(|c| c.into_iter().collect()).collect(),
    };
    if parity {
        out.terms.push(Vec::new());
    }
    out.sort_terms();
    debug_assert!(out.terms.len() <= g.terms.len());
    out
}

pub fn compact(packed: &[PackedGate]) -> Vec<PackedGate> {
    packed.iter().map(compact_gate).collect()
}

/// Pack every maximal same-target run into one canonical ANF gate. Exact for
/// any gate list; canonical (one representation per activation function) and
/// maximally packed when the list is a gather fixed point, i.e. fcompress
/// output.
pub fn pack(gates: &[XGate]) -> Vec<PackedGate> {
    let mut out = Vec::new();
    let mut i = 0usize;
    while i < gates.len() {
        let mut j = i + 1;
        while j < gates.len() && gates[j].target == gates[i].target {
            j += 1;
        }
        out.push(pack_run(&gates[i..j]));
        i = j;
    }
    out
}

// Packing census: what packing leaves and what the canonical form costs.
// Per run: current cube count, ANF monomials, support, monomial degrees;
// plus the size of the deterministic canonical ESOP (anf_reduce applied to
// the ANF, support <= 63 only). Prints a few lines.
pub fn pack_census(gates: &[XGate]) {
    fn bin(x: usize) -> usize {
        match x {
            0 => 0,
            1 => 1,
            2 => 2,
            3..=4 => 3,
            5..=8 => 4,
            9..=16 => 5,
            17..=32 => 6,
            33..=64 => 7,
            65..=256 => 8,
            _ => 9,
        }
    }
    const LABELS: [&str; 10] = [
        "0", "1", "2", "3-4", "5-8", "9-16", "17-32", "33-64", "65-256", ">256",
    ];
    let mut cubes_h = [0usize; 10];
    let mut anf_h = [0usize; 10];
    let mut sup_h = [0usize; 10];
    let mut canon_h = [0usize; 10];
    let mut deg_h = [0usize; 10];
    let (mut runs, mut multi_runs, mut multi_mass) = (0usize, 0usize, 0usize);
    let (mut anf_total, mut anf_max, mut sup_max) = (0usize, 0usize, 0usize);
    let (mut cube_lits, mut mono_degs) = (0usize, 0usize);
    let (mut blowup_runs, mut blowup_extra) = (0usize, 0usize);
    let (mut canon_total, mut canon_cubes, mut canon_skipped) = (0usize, 0usize, 0usize);
    let (mut canon_smaller, mut canon_larger, mut canon_gain, mut canon_loss) =
        (0usize, 0usize, 0usize, 0usize);
    let mut i = 0usize;
    while i < gates.len() {
        let mut j = i + 1;
        while j < gates.len() && gates[j].target == gates[i].target {
            j += 1;
        }
        let run = &gates[i..j];
        let k = run.len();
        runs += 1;
        cubes_h[bin(k)] += 1;
        if k > 1 {
            multi_runs += 1;
            multi_mass += k;
        }
        cube_lits += run.iter().map(XGate::width).sum::<usize>();
        let pg = pack_run(run);
        let m = pg.terms.len();
        anf_h[bin(m)] += 1;
        anf_total += m;
        anf_max = anf_max.max(m);
        let mut support: Vec<u16> = pg.terms.iter().flatten().map(|l| l.0).collect();
        support.sort_unstable();
        support.dedup();
        sup_h[bin(support.len())] += 1;
        sup_max = sup_max.max(support.len());
        for mo in &pg.terms {
            mono_degs += mo.len();
            deg_h[bin(mo.len())] += 1;
        }
        if m > k {
            blowup_runs += 1;
            blowup_extra += m - k;
        }
        if support.len() <= 63 {
            let canon = compact_gate(&pg).terms.len();
            canon_h[bin(canon)] += 1;
            canon_total += canon;
            canon_cubes += k;
            if canon < k {
                canon_smaller += 1;
                canon_gain += k - canon;
            } else if canon > k {
                canon_larger += 1;
                canon_loss += canon - k;
            }
        } else {
            canon_skipped += 1;
        }
        i = j;
    }
    let g = gates.len();
    println!(
        "[pack] gates={} runs(=packed gates)={} saved={} ({:.1}%) | multi runs={} ({:.1}% of runs) holding {} gates ({:.1}% of mass)",
        g,
        runs,
        g - runs,
        100.0 * (g - runs) as f64 / g.max(1) as f64,
        multi_runs,
        100.0 * multi_runs as f64 / runs.max(1) as f64,
        multi_mass,
        100.0 * multi_mass as f64 / g.max(1) as f64
    );
    println!(
        "[pack] canonical ANF: {} monomials for {} cubes (x{:.2}); runs where ANF > cubes: {} (+{} monomials); max monomials in a run: {}; max support: {} wires",
        anf_total,
        g,
        anf_total as f64 / g.max(1) as f64,
        blowup_runs,
        blowup_extra,
        anf_max,
        sup_max
    );
    println!(
        "[pack] description mass: cubes carry {} literals, ANF carries {} wire occurrences (x{:.2}); mean cube width {:.2}, mean monomial degree {:.2}",
        cube_lits,
        mono_degs,
        mono_degs as f64 / cube_lits.max(1) as f64,
        cube_lits as f64 / g.max(1) as f64,
        mono_degs as f64 / anf_total.max(1) as f64
    );
    println!(
        "[pack] compacted ESOP (deterministic, from the ANF alone, support<=63; {} runs skipped): {} terms vs {} cubes now; smaller on {} runs (-{}), larger on {} runs (+{})",
        canon_skipped,
        canon_total,
        canon_cubes,
        canon_smaller,
        canon_gain,
        canon_larger,
        canon_loss
    );
    let show = |name: &str, h: &[usize; 10]| {
        let parts: Vec<String> = LABELS
            .iter()
            .zip(h.iter())
            .filter(|(_, c)| **c > 0)
            .map(|(l, c)| format!("{l}:{c}"))
            .collect();
        println!("[pack] {name}: {}", parts.join(" "));
    };
    show("cubes per run", &cubes_h);
    show("ANF monomials per run", &anf_h);
    show("monomial degree", &deg_h);
    show("compacted ESOP terms per run", &canon_h);
    show("support wires per run", &sup_h);
}
