//! Scan-only leakage-detector runs: hot-segment statistics of mixed circuits
//! against a fixed reference, NO repairs. The detector is the production QC
//! one (leakage_repair::detect); this tool just runs `scan` on each input and
//! prints one row per circuit, so hot-segment counts can be tracked along a
//! mixing trajectory ("how many wire segments would prompt a QC splice against
//! the stage-2 output at eff k").
//!
//! Hot = an internal value segment of the mixed circuit that nearly determines
//! a sampled reference feature (|phi| >= min_abs_correlation in BOTH sample
//! partitions, or an exact affine train fit validating at 100% on held-out).
//! `hot_count` is uncapped; `rate` = hot/scanned; `est_total` = rate *
//! internal_segments (the scan samples segments when the circuit has more
//! than --mixed-segments of them).
//!
//! ```text
//! qc_scan REFERENCE.mpmct1 MIXED.mpmct1 [MIXED ...]
//!     [--mixed-segments N=4096] [--orig-segments N=128] [--orig-firings N=256]
//!     [--seed S]
//! ```

use local_mixing::circuit::formats::read_mpmct;
use local_mixing::stages::db_mixing::leakage_repair::detect::{Detector, DetectorConfig};
use std::error::Error;

type AnyResult<T> = Result<T, Box<dyn Error + Send + Sync>>;

fn main() {
    if let Err(error) = run() {
        eprintln!("qc_scan: {error}");
        std::process::exit(1);
    }
}

fn run() -> AnyResult<()> {
    let args: Vec<String> = std::env::args().collect();
    if args.len() < 3 {
        eprintln!(
            "usage: qc_scan REFERENCE.mpmct1 MIXED.mpmct1 [MIXED ...] \
             [--mixed-segments N] [--orig-segments N] [--orig-firings N] [--seed S]"
        );
        std::process::exit(2);
    }
    let flag = |name: &str| {
        args.iter()
            .position(|a| a == name)
            .and_then(|i| args.get(i + 1))
            .and_then(|s| s.parse::<usize>().ok())
    };
    let mixed_segments = flag("--mixed-segments").unwrap_or(4096);
    let orig_segments = flag("--orig-segments").unwrap_or(128);
    let orig_firings = flag("--orig-firings").unwrap_or(256);
    let seed = flag("--seed").unwrap_or(0x5143) as u64;
    let paths: Vec<&String> = args[1..]
        .iter()
        .take_while(|a| !a.starts_with("--"))
        .collect();
    let (reference, ref_wires) =
        read_mpmct(paths[0]).map_err(|e| format!("read {}: {e}", paths[0]))?;

    let mut config = DetectorConfig::default();
    config.seed = seed;
    config.max_mixed_segments = mixed_segments;
    config.max_hot_segments = mixed_segments;
    config.max_original_segments = orig_segments;
    config.max_original_firings = orig_firings;
    config.max_trace_words = 1_073_741_824; // 8 GiB of u64s: roomy budget

    let detector = Detector::new(&reference, ref_wires as usize, config)
        .map_err(|e| format!("detector: {e}"))?;
    println!(
        "{:>44} {:>9} {:>11} {:>8} {:>7} {:>7} {:>10} {:>8} {:>6}",
        "circuit", "gates", "int_segs", "scanned", "hot", "rate%", "est_total", "hot_bnd", "trunc"
    );
    let dump_hot = args.iter().any(|a| a == "--dump-hot");
    let hot_out = args
        .iter()
        .position(|a| a == "--hot-out")
        .and_then(|i| args.get(i + 1));
    for path in &paths[1..] {
        let (mixed, _) = read_mpmct(path).map_err(|e| format!("read {path}: {e}"))?;
        let report = detector
            .scan(&mixed)
            .map_err(|e| format!("scan {path}: {e}"))?;
        let rate = report.hot_segment_count as f64 / report.scanned_segments.max(1) as f64;
        let name = path.rsplit('/').next().unwrap_or(path);
        println!(
            "{:>44} {:>9} {:>11} {:>8} {:>7} {:>6.2}% {:>10.0} {:>8} {:>6}",
            name,
            mixed.len(),
            report.internal_segments,
            report.scanned_segments,
            report.hot_segment_count,
            100.0 * rate,
            rate * report.internal_segments as f64,
            report.hot_boundary_firings,
            u8::from(report.truncated) + 2 * u8::from(report.reference_truncated),
        );
        if let Some(out) = hot_out {
            use std::io::Write as _;
            let mut f = std::fs::File::create(out).map_err(|e| format!("{out}: {e}"))?;
            writeln!(f, "# wire\tstart_gate\tend_gate (hot segments of {path})")
                .map_err(|e| e.to_string())?;
            for h in &report.hot_segments {
                writeln!(f, "{}\t{}\t{}", h.wire, h.start_gate, h.end_gate)
                    .map_err(|e| e.to_string())?;
            }
            eprintln!("wrote {} hot segments to {out}", report.hot_segments.len());
        }
        if dump_hot {
            use local_mixing::stages::db_mixing::leakage_repair::detect::Evidence;
            // Mixed side: where the carriers sit. Reference side: which
            // producer gates the matched features come from -- their position
            // decile in the reference and their target-wire class.
            let n = mixed.len().max(1);
            let rn = reference.len().max(1);
            let mut decile = [0usize; 10];
            let mut ref_decile = [0usize; 10];
            let mut ref_wires = [0usize; 3]; // low 0..64, payload 64..128, high 128..
            for h in &report.hot_segments {
                let mid = (h.start_gate + h.end_gate) / 2;
                decile[(10 * mid / n).min(9)] += 1;
                for e in &h.evidence {
                    let gates: Vec<usize> = match e {
                        Evidence::Affine {
                            original_features, ..
                        } => original_features.clone(),
                        Evidence::FiringCorrelation { original_gate, .. } => {
                            vec![*original_gate]
                        }
                    };
                    for g in gates {
                        if g < reference.len() {
                            ref_decile[(10 * g / rn).min(9)] += 1;
                            let t = reference[g].target;
                            ref_wires[if t < 64 {
                                0
                            } else if t < 128 {
                                1
                            } else {
                                2
                            }] += 1;
                        }
                    }
                }
            }
            println!("    mixed-side deciles:     {decile:?}");
            println!("    reference-side deciles: {ref_decile:?}");
            println!(
                "    reference wires: low(0..63)={} payload(64..127)={} band(128..255)={}",
                ref_wires[0], ref_wires[1], ref_wires[2]
            );
        }
    }
    Ok(())
}
