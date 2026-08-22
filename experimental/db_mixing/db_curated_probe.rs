//! Survey: how much of the curated store is reachable, and is it correct?
//!
//! The curated store is keyed on the FORWARD canonical form only -- the regular
//! fallback is what may also try the reversed one. A lookup that asks curated in
//! the reverse frame gets back entries belonging to a different permutation, so
//! the useful question is not "is curated correct" but "does curated answer
//! forward-frame questions often enough to be worth having".
//!
//! Samples contiguous windows from a circuit, decodes EVERY candidate from both
//! stores in both directions, and tallies reachability and equivalence
//! separately for each (store, direction).
//!
//! Usage: db_curated_probe <circuit.mpmct1> [samples] [max_window]
use local_mixing::db_mixing::db_replace::db_probe;
use local_mixing::engine::format;
use local_mixing::engine::rules::verify_rewrite;
use local_mixing::circuit::xgate::XGate;
use local_mixing::engine::xpoly::XPolyBudget;
use local_mixing::db_mixing::frozen::FrozenDb;
use rand::rngs::StdRng;
use rand::{Rng, SeedableRng};

fn main() {
    let mut a = std::env::args().skip(1);
    let path = a.next().expect("usage: db_curated_probe <circuit> [samples] [max_window]");
    let samples: usize = a.next().and_then(|s| s.parse().ok()).unwrap_or(2000);
    let wmax: usize = a.next().and_then(|s| s.parse().ok()).unwrap_or(5);
    let (gates, num_wires) = format::read_mpmct(&path).expect("read circuit");
    println!("circuit {path}: {} gates, {num_wires} wires; {samples} windows up to {wmax}", gates.len());

    let db = FrozenDb::from_env();
    let budget = XPolyBudget::default();
    let mut rng = StdRng::seed_from_u64(7);

    let (mut probed, mut skipped) = (0u64, 0u64);
    let (mut w_reg, mut w_cur_fwd, mut w_cur_rev) = (0u64, 0u64, 0u64);
    let (mut cf_ok, mut cf_bad, mut cr_ok, mut cr_bad) = (0u64, 0u64, 0u64, 0u64);
    let (mut r_ok, mut r_bad) = (0u64, 0u64);

    while probed < samples as u64 {
        let len = rng.random_range(2..=wmax);
        if gates.len() < len {
            break;
        }
        let at = rng.random_range(0..gates.len() - len);
        let win: Vec<XGate> = gates[at..at + len].to_vec();
        // Same guards the walk applies: no wide gates, bounded support.
        if win.iter().any(|g| g.ctrls.len() >= 4) {
            skipped += 1;
            continue;
        }
        if local_mixing::engine::xpoly::xgate_used_wires(&win).len() > 30 {
            skipped += 1;
            continue;
        }
        probed += 1;
        let cands = db_probe(&win, num_wires, &db, budget, &mut rng);
        let (mut has_reg, mut has_cf, mut has_cr) = (false, false, false);
        for (g, curated, reversed) in &cands {
            let ok = verify_rewrite(&win, g);
            match (curated, reversed, ok) {
                (true, false, true) => { cf_ok += 1; has_cf = true }
                (true, false, false) => { cf_bad += 1; has_cf = true }
                (true, true, true) => { cr_ok += 1; has_cr = true }
                (true, true, false) => { cr_bad += 1; has_cr = true }
                (false, _, true) => { r_ok += 1; has_reg = true }
                (false, _, false) => { r_bad += 1; has_reg = true }
            }
        }
        w_reg += has_reg as u64;
        w_cur_fwd += has_cf as u64;
        w_cur_rev += has_cr as u64;
    }

    let pct = |x: u64| 100.0 * x as f64 / probed.max(1) as f64;
    println!("\nwindows probed {probed} (skipped {skipped} for width/span)");
    println!("  with any REGULAR candidate:        {w_reg} ({:.1}%)", pct(w_reg));
    println!("  with any CURATED FORWARD candidate:{w_cur_fwd} ({:.1}%)", pct(w_cur_fwd));
    println!("  with any CURATED REVERSE candidate:{w_cur_rev} ({:.1}%)", pct(w_cur_rev));
    println!("\ncandidate equivalence (this is the correctness question):");
    println!("  regular          : {r_ok} ok, {r_bad} BAD");
    println!("  curated forward  : {cf_ok} ok, {cf_bad} BAD");
    println!("  curated reverse  : {cr_ok} ok, {cr_bad} BAD   <- never queried in production");
}
