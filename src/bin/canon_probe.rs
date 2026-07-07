// Throwaway: characterize WHERE canonicalization spends time on a given window.
// Loads a circuit, canonicalizes it forward (the exact production path), and reports
// elapsed time alongside the Rule-L counters, so we can tell a Rule-L blowup from a
// tiebreak-loop or polynomial-build blowup. Respects CANON_MONOMIAL_CAP.
// Usage: canon_probe <circuit.txt>
use local_mixing::circuit::CircuitSeq;
use local_mixing::circuit::circuit::{CANON4_RULE_L_BRANCHES, CANON4_RULE_L_CALLS, CANON4_RULE_L_TIME};
use std::sync::atomic::Ordering::Relaxed;
use std::time::Instant;

fn main() {
    let path = std::env::args().nth(1).expect("usage: canon_probe <circuit.txt>");
    let s = std::fs::read_to_string(&path).expect("read circuit");
    let c = CircuitSeq::from_string(&s);
    eprintln!("loaded {} gates, {} used wires", c.gates.len(), c.used_wires().len());

    let c0 = CANON4_RULE_L_CALLS.load(Relaxed);
    let b0 = CANON4_RULE_L_BRANCHES.load(Relaxed);
    let t0 = CANON4_RULE_L_TIME.load(Relaxed);

    // Live watcher: print Rule-L branch growth each second so a hang shows its cause.
    std::thread::spawn(move || {
        loop {
            std::thread::sleep(std::time::Duration::from_secs(1));
            eprintln!(
                "[live] rule_l_calls={} rule_l_branches={}",
                CANON4_RULE_L_CALLS.load(Relaxed) - c0,
                CANON4_RULE_L_BRANCHES.load(Relaxed) - b0
            );
        }
    });

    let start = Instant::now();
    let (polys, _order, used) = c.canonicalize_polys_single(false);
    let elapsed = start.elapsed();

    let calls = CANON4_RULE_L_CALLS.load(Relaxed) - c0;
    let branches = CANON4_RULE_L_BRANCHES.load(Relaxed) - b0;
    let rl_time_ns = CANON4_RULE_L_TIME.load(Relaxed) - t0;

    println!("total_elapsed_ms={}", elapsed.as_millis());
    println!("rule_l_calls={} rule_l_branches={} rule_l_time_ms={}", calls, branches, rl_time_ns / 1_000_000);
    println!("rule_l_fraction_of_total={:.3}", rl_time_ns as f64 / elapsed.as_nanos().max(1) as f64);
    println!("result_polys={} (empty={}) used_wires={}", polys.len(), polys.is_empty(), used.len());
}
