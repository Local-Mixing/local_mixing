//! dmr_probe — feasibility + correctness probe for the DMR-k relay
//! (Deep-Masked k-Site Relay, docs/NONLOCAL_PHASE_A.md design notes).
//!
//! Store-INDEPENDENT. Answers the two questions the fleet cannot: (1) does the
//! k-control deep-masked relay actually telescope to identity on real gates
//! (function preservation), and (2) is it feasible on real material — can we
//! find an idle carry lane w and a deep interior wire g within the degree/wire
//! budget, and how big is the wake / how often does it refuse?
//!
//! It does NOT touch the frozen store (that hit-rate question needs the fleet);
//! it reports the endpoint-window degree/span so we know whether the windows
//! COULD pass the store guards (degree <= 9, span <= 30).
//!
//! The relay (verified here by evaluation):
//!   a · M · e   ->   a · D · Mconj · F · e · Ecorr · F · D
//! with D = (w ^= phi_a ∧ g)  [deposit, phi_a = a's control monomial, g deep],
//!      Mconj = each interior h replaced by [h, conj_wake_k(D,h)]  (= D·h·D),
//!      F = (d ^= w)  [fold, d a control of e],
//!      Ecorr = conj_wake_k(F, e)  [uncompute e's absorbed perturbation].
//! Telescoping: D·(D·M·D) = M·D, and D·F·e·Ecorr·F·D = e, so the whole thing
//! equals a·M·e while t_e's TRACE at e's index carried c∧phi_a∧g.

use clap::Parser;
use local_mixing::circuit::xgate::XGate;
use local_mixing::engine::format::read_mpmct;
use rand::rngs::StdRng;
use rand::{Rng, SeedableRng};

#[derive(Parser)]
struct Args {
    #[arg(long, default_value = "")]
    input: String,
    #[arg(long, default_value = "mpmct1")]
    g_format: String,
    #[arg(long, default_value_t = 20000)]
    samples: usize,
    #[arg(long, default_value_t = 8)]
    min_span: usize,
    #[arg(long, default_value_t = 512)]
    max_span: usize,
    #[arg(long, default_value_t = 12)]
    k_max: usize,
    /// Random-input batches (×64 lanes) for the per-relay correctness check.
    #[arg(long, default_value_t = 4)]
    check_batches: usize,
    #[arg(long, default_value_t = 1)]
    seed: u64,
    /// Run the exhaustive small-circuit self-test (all 2^n inputs) instead of
    /// probing a file — proves the relay algebra exactly.
    #[arg(long, default_value_t = false)]
    selftest: bool,
}

// Exhaustive functional equality over wires 0..n (n <= 16), 64 lanes at a time.
fn eq_exhaustive(a: &[XGate], b: &[XGate], n: usize) -> bool {
    let total: u64 = 1 << n;
    let mut v = 0u64;
    while v < total {
        let mut sa = vec![0u64; n];
        for (i, wv) in sa.iter_mut().enumerate() {
            let mut acc = 0u64;
            for l in 0..64u64 {
                if ((v + l) >> i) & 1 == 1 {
                    acc |= 1 << l;
                }
            }
            *wv = acc;
        }
        let mut sb = sa.clone();
        for g in a {
            g.apply_lanes(&mut sa);
        }
        for g in b {
            g.apply_lanes(&mut sb);
        }
        let valid: u64 = if total - v >= 64 {
            !0
        } else {
            (1u64 << (total - v)) - 1
        };
        for i in 0..n {
            if (sa[i] ^ sb[i]) & valid != 0 {
                return false;
            }
        }
        v += 64;
    }
    true
}

fn selftest() {
    let n = 12usize;
    let mut rng = StdRng::seed_from_u64(0xd3f);
    let (mut ok, mut feasible, mut broken) = (0u32, 0u32, 0u32);
    for _ in 0..4000 {
        // Random small circuit of g57s + conjunctions.
        let m = rng.random_range(6..14);
        let gates: Vec<XGate> = (0..m)
            .map(|_| {
                let t = rng.random_range(0..n as u16);
                let mut a = rng.random_range(0..n as u16);
                let mut b = rng.random_range(0..n as u16);
                while a == t {
                    a = rng.random_range(0..n as u16);
                }
                while b == t || b == a {
                    b = rng.random_range(0..n as u16);
                }
                XGate::from_g57([t, a, b])
            })
            .collect();
        let i_a = rng.random_range(0..m - 2);
        let i_e = rng.random_range(i_a + 1..m);
        let a = &gates[i_a];
        let e = &gates[i_e];
        // pick w, g among wires not in the span-written / endpoint set
        let mut written = vec![false; n];
        for h in &gates[i_a + 1..=i_e] {
            written[h.target as usize] = true;
        }
        let mut excl = vec![false; n];
        for &(x, _) in a.ctrls.iter().chain(e.ctrls.iter()) {
            excl[x as usize] = true;
        }
        excl[a.target as usize] = true;
        excl[e.target as usize] = true;
        let free: Vec<u16> = (0..n as u16)
            .filter(|&x| !written[x as usize] && !excl[x as usize])
            .collect();
        if free.len() < 2 {
            continue;
        }
        let w = free[0];
        let g = free[1];
        if let Some(r) = build_relay(&gates, i_a, i_e, w, g, true, 12) {
            feasible += 1;
            let mut relayed = gates[..i_a].to_vec();
            relayed.extend(r.seq.iter().cloned());
            relayed.extend(gates[i_e + 1..].iter().cloned());
            if eq_exhaustive(&gates, &relayed, n) {
                ok += 1;
            } else {
                broken += 1;
            }
        }
    }
    println!(
        "[dmr_probe selftest] feasible={feasible} exact={ok} BROKEN={broken} (broken must be 0)"
    );
    assert_eq!(
        broken, 0,
        "relay algebra broke function on an exhaustive check"
    );
}

// ---- k-control conjugation wake (generalizes src/engine/mix.rs conj_wake to
// a comp0 carrier u with ANY number of controls; monomial m = u.ctrls) ----

enum Merged {
    Gate(XGate),
    Never,   // contradictory literal: correction identically 0
    Invalid, // target among controls, or wider than k_max
}

fn merged_conj(target: u16, lits: &[(u16, bool)], k_max: usize) -> Merged {
    let mut merged: Vec<(u16, bool)> = Vec::with_capacity(lits.len());
    for &(w, p) in lits {
        if w == target {
            return Merged::Invalid;
        }
        match merged.iter().find(|&&(mw, _)| mw == w) {
            Some(&(_, mp)) if mp != p => return Merged::Never,
            Some(_) => {}
            None => merged.push((w, p)),
        }
    }
    if merged.len() > k_max {
        return Merged::Invalid;
    }
    match XGate::conj(target, merged) {
        Some(g) => Merged::Gate(g),
        None => Merged::Never,
    }
}

/// Corrections C so that [h] ++ C computes exactly u·h·u. `u` must be comp0.
/// None = refuse (mutual collision or a correction wider than k_max).
fn conj_wake_k(u: &XGate, h: &XGate, k_max: usize) -> Option<Vec<XGate>> {
    debug_assert!(!u.comp, "carrier must be a conjunction (comp0)");
    let tu = u.target;
    let reads_tu = h.reads(tu);
    let writes_m = u.reads(h.target); // h.target is one of u's control wires
    if !reads_tu && !writes_m {
        return Some(Vec::new());
    }
    if reads_tu && writes_m {
        return None;
    }
    let mut corrs: Vec<XGate> = Vec::new();
    if reads_tu {
        // corr = (t_h ; m ∧ L),  m = u.ctrls,  L = h's literals except on tu.
        let lits: Vec<(u16, bool)> = u
            .ctrls
            .iter()
            .copied()
            .chain(h.ctrls.iter().copied().filter(|&(w, _)| w != tu))
            .collect();
        match merged_conj(h.target, &lits, k_max) {
            Merged::Gate(c) => corrs.push(c),
            Merged::Never => {}
            Merged::Invalid => return None,
        }
    } else {
        // h writes control wire h.target of u; rho = u's OTHER literals.
        let rho: Vec<(u16, bool)> = u
            .ctrls
            .iter()
            .copied()
            .filter(|&(w, _)| w != h.target)
            .collect();
        if h.comp {
            match merged_conj(tu, &rho, k_max) {
                Merged::Gate(c) => corrs.push(c),
                Merged::Never => {}
                Merged::Invalid => return None,
            }
        }
        let lits: Vec<(u16, bool)> = h.ctrls.iter().copied().chain(rho.iter().copied()).collect();
        match merged_conj(tu, &lits, k_max) {
            Merged::Gate(c) => corrs.push(c),
            Merged::Never => {}
            Merged::Invalid => return None,
        }
    }
    Some(corrs)
}

struct Relay {
    seq: Vec<XGate>,   // the replacement for the [a ..= e] segment
    wake: usize,       // correction gates emitted across the interior
    dep_wires: usize,  // distinct wires in the deposit window [a, D]
    dep_deg: usize,    // ANF degree bound of the deposit window
    fold_wires: usize, // distinct wires in [F, e, Ecorr]
    fold_deg: usize,
    mu_deg: usize, // degree of the deposited monomial phi_a ∧ g
}

fn distinct_wires(gates: &[XGate]) -> usize {
    let mut ws: Vec<u16> = gates
        .iter()
        .flat_map(|g| std::iter::once(g.target).chain(g.ctrls.iter().map(|&(w, _)| w)))
        .collect();
    ws.sort_unstable();
    ws.dedup();
    ws.len()
}

/// Build the relay for segment gates[i_a ..= i_e]; None on any refusal.
fn build_relay(
    gates: &[XGate],
    i_a: usize,
    i_e: usize,
    w: u16,
    g: u16,
    gpol: bool,
    k_max: usize,
) -> Option<Relay> {
    let a = &gates[i_a];
    let e = &gates[i_e];
    if a.ctrls.is_empty() || e.ctrls.is_empty() {
        return None;
    }
    let d = e.ctrls[0].0; // fold control
    // Deposit carrier D = (w ; a.ctrls ∧ g). a is g57/conjunction; its "firing
    // monomial" is a.ctrls (the comp bit rides on the kept a, not the carry).
    let mut dep_lits: Vec<(u16, bool)> = a.ctrls.iter().copied().collect();
    if dep_lits.iter().any(|&(x, _)| x == w || x == g) || a.target == w {
        return None;
    }
    // The collector D must commute with e (the tail telescopes as D·e·D = e).
    // D reads a.ctrls ∪ {g}; it collides with e iff e's TARGET is one of those
    // wires. g != t_e by selection; guard a.ctrls here.
    if a.ctrls.iter().any(|&(x, _)| x == e.target) {
        return None;
    }
    dep_lits.push((g, gpol));
    let big_d = XGate::conj(w, dep_lits.clone())?;
    // Fold F = (d ^= w); Ecorr = conj_wake_k(F, e).
    let f = XGate::conj(d, [(w, true)])?;
    let ecorr = conj_wake_k(&f, e, k_max)?;

    // Interior conjugation: each h -> [h, conj_wake_k(D, h)].
    let mut mconj: Vec<XGate> = Vec::new();
    let mut wake = 0usize;
    for h in &gates[i_a + 1..i_e] {
        let corrs = conj_wake_k(&big_d, h, k_max)?;
        wake += corrs.len();
        mconj.push(h.clone());
        mconj.extend(corrs);
    }

    // Assemble: a, D, Mconj, F, e, Ecorr, F, D.
    let mut seq: Vec<XGate> = Vec::new();
    seq.push(a.clone());
    seq.push(big_d.clone());
    seq.extend(mconj);
    seq.push(f.clone());
    seq.push(e.clone());
    seq.extend(ecorr.clone());
    seq.push(f.clone());
    seq.push(big_d.clone());

    let dep_win = [a.clone(), big_d.clone()];
    let mut fold_win = vec![f.clone(), e.clone()];
    fold_win.extend(ecorr);
    Some(Relay {
        seq,
        wake,
        dep_wires: distinct_wires(&dep_win),
        dep_deg: a.ctrls.len() + 1, // w-output monomial has |a.ctrls|+1 literals
        fold_wires: distinct_wires(&fold_win),
        fold_deg: e.ctrls.len() + 1,
        mu_deg: dep_lits.len(),
    })
}

/// True iff the two gate segments compute the same function, tested on
/// `batches`×64 random full-width inputs.
fn segments_equal(
    orig: &[XGate],
    relay: &[XGate],
    num_wires: usize,
    batches: usize,
    rng: &mut StdRng,
) -> bool {
    for _ in 0..batches {
        let init: Vec<u64> = (0..num_wires).map(|_| rng.random::<u64>()).collect();
        let mut s1 = init.clone();
        let mut s2 = init;
        for g in orig {
            g.apply_lanes(&mut s1);
        }
        for g in relay {
            g.apply_lanes(&mut s2);
        }
        if s1 != s2 {
            return false;
        }
    }
    true
}

fn main() {
    let args = Args::parse();
    if args.selftest {
        selftest();
        return;
    }
    let (gates, num_wires) = match args.g_format.as_str() {
        "mpmct1" => read_mpmct(&args.input).expect("read mpmct1"),
        o => panic!("unknown --g-format {o} (this probe reads mpmct1)"),
    };
    let n = gates.len();
    eprintln!("[dmr_probe] loaded {n} gates on {num_wires} wires");
    let mut rng = StdRng::seed_from_u64(args.seed);

    let mut attempts = 0u64;
    let mut no_endpoint = 0u64; // a or e had no controls
    let mut no_carry = 0u64; // no idle carry wire
    let mut no_deep = 0u64; // no deep interior wire
    let mut refused = 0u64; // conj_wake_k returned None somewhere
    let mut feasible = 0u64;
    let mut correct = 0u64;
    let mut broken = 0u64; // feasible but function NOT preserved (a bug!)
    let mut guard_ok = 0u64; // feasible AND both windows pass store guards
    let mut wake_sum = 0u64;
    let mut wake_max = 0u64;
    let mut deg_max = 0u64;
    let mut wires_max = 0u64;
    let mut deep_reads_sum = 0u64;

    for _ in 0..args.samples {
        attempts += 1;
        if n < args.min_span + 2 {
            break;
        }
        let i_a = rng.random_range(0..n - args.min_span - 1);
        // log-uniform span
        let lo = args.min_span as f64;
        let hi = (args.max_span.min(n - i_a - 1)).max(args.min_span) as f64;
        let span = (lo * (hi / lo).powf(rng.random::<f64>())).round() as usize;
        let i_e = i_a + span.clamp(args.min_span, args.max_span);
        if i_e >= n {
            continue;
        }
        if gates[i_a].ctrls.is_empty() || gates[i_e].ctrls.is_empty() {
            no_endpoint += 1;
            continue;
        }
        // Census the span (i_a, i_e]: written wires and per-wire read counts.
        let mut written = vec![false; num_wires];
        let mut reads = vec![0u32; num_wires];
        for h in &gates[i_a + 1..=i_e] {
            written[h.target as usize] = true;
            for &(x, _) in &h.ctrls {
                reads[x as usize] += 1;
            }
        }
        // Excluded wires: endpoints' support.
        let mut excluded = vec![false; num_wires];
        for &(x, _) in &gates[i_a].ctrls {
            excluded[x as usize] = true;
        }
        excluded[gates[i_a].target as usize] = true;
        for &(x, _) in &gates[i_e].ctrls {
            excluded[x as usize] = true;
        }
        excluded[gates[i_e].target as usize] = true;

        // Idle carry w: unwritten in span, not excluded, least-read.
        let mut w_pick: Option<u16> = None;
        let mut w_reads = u32::MAX;
        // Deep wire g: unwritten in span, not excluded, MOST-read (deep proxy).
        let mut g_pick: Option<u16> = None;
        let mut g_reads = 0u32;
        for wire in 0..num_wires {
            if written[wire] || excluded[wire] {
                continue;
            }
            if reads[wire] < w_reads {
                w_reads = reads[wire];
                w_pick = Some(wire as u16);
            }
            if reads[wire] >= g_reads {
                // deep must be actually read (>0) to be a meaningful predicate
                g_reads = reads[wire];
                g_pick = Some(wire as u16);
            }
        }
        let Some(w) = w_pick else {
            no_carry += 1;
            continue;
        };
        let Some(g) = g_pick else {
            no_deep += 1;
            continue;
        };
        if g == w || g_reads == 0 {
            no_deep += 1;
            continue;
        }

        match build_relay(&gates, i_a, i_e, w, g, true, args.k_max) {
            None => {
                refused += 1;
            }
            Some(r) => {
                feasible += 1;
                wake_sum += r.wake as u64;
                wake_max = wake_max.max(r.wake as u64);
                deg_max = deg_max.max(r.dep_deg.max(r.fold_deg) as u64);
                wires_max = wires_max.max(r.dep_wires.max(r.fold_wires) as u64);
                deep_reads_sum += g_reads as u64;
                // Store-guard check on both endpoint windows.
                if r.dep_deg <= 9 && r.fold_deg <= 9 && r.dep_wires <= 30 && r.fold_wires <= 30 {
                    guard_ok += 1;
                }
                // Correctness: the relay must equal the original segment.
                let orig = &gates[i_a..=i_e];
                if segments_equal(orig, &r.seq, num_wires, args.check_batches, &mut rng) {
                    correct += 1;
                } else {
                    broken += 1;
                    if broken <= 3 {
                        eprintln!(
                            "[dmr_probe] !!! FUNCTION NOT PRESERVED at i_a={i_a} i_e={i_e} w={w} g={g} (mu_deg={})",
                            r.mu_deg
                        );
                    }
                }
            }
        }
    }

    let fe = feasible.max(1) as f64;
    println!("[dmr_probe] attempts={attempts}");
    println!(
        "[dmr_probe] rejected: no_endpoint={no_endpoint} no_carry={no_carry} no_deep={no_deep} refused(conj_wake)={refused}"
    );
    println!(
        "[dmr_probe] FEASIBLE={feasible} ({:.1}% of attempts)",
        100.0 * feasible as f64 / attempts.max(1) as f64
    );
    println!("[dmr_probe] CORRECTNESS: preserved={correct} BROKEN={broken}  (broken must be 0)");
    println!(
        "[dmr_probe] store-guard-pass (deg<=9 & wires<=30 both windows)={guard_ok} ({:.1}% of feasible)",
        100.0 * guard_ok as f64 / fe
    );
    println!(
        "[dmr_probe] wake: mean={:.2} max={} | window degree max={} wires max={} | deep-g mean reads={:.1}",
        wake_sum as f64 / fe,
        wake_max,
        deg_max,
        wires_max,
        deep_reads_sum as f64 / fe,
    );
}
