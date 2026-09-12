//! Historical environment adapter for blinded-V5 comparison tools.
//! New library callers use `stages::preprocessing::quadratic_masking` with explicit options.

use crate::circuit::xgate::XGate;
use crate::stages::preprocessing::quadratic_masking::{
    QuadraticMaskingExecution, preprocess_quadratic_masking_with_execution,
};
pub use crate::stages::preprocessing::quadratic_masking::{
    QuadraticMaskingOutput as BlindedV5Output, QuadraticMaskingParams as BlindedV5Params,
    hot_intervals, seed_band, seed_band_mode,
};

pub fn gadgetize_blinded_v5(src: &[XGate], np: usize, p: &BlindedV5Params) -> BlindedV5Output {
    let execution = QuadraticMaskingExecution {
        ancilla_band_only: std::env::var("BV5_ANC_POOL").map_or(true, |v| v != "all"),
        diagnostics: std::env::var("BV5_DIAG").is_ok(),
    };
    let output = preprocess_quadratic_masking_with_execution(src, np, p, &execution);
    write_hot_manifest_from_env(&output, np);
    output
}

/// Retained optional artifact adapter; it never changes circuit construction.
pub fn write_hot_manifest_from_env(output: &BlindedV5Output, np: usize) {
    if let Ok(path) = std::env::var("BV5_HOT_MANIFEST") {
        let hot = hot_intervals(&output.gates, np, output.r_used);
        let mut body = String::from("# wire\tstart_gate\tend_gate\tkind\n");
        let (mut fringe, mut interior) = (0usize, 0usize);
        for &(w, a, b) in &hot {
            let kind = if a == 0 {
                fringe += 1;
                "input-fringe"
            } else if b == output.gates.len() {
                fringe += 1;
                "output-fringe"
            } else {
                interior += 1;
                "INTERIOR (defect)"
            };
            body.push_str(&format!("{w}\t{a}\t{b}\t{kind}\n"));
        }
        if std::fs::write(&path, body).is_ok() {
            eprintln!(
                "[bv5-hot] {} affine intervals ({fringe} I/O fringe, {interior} interior) -> {path}",
                hot.len()
            );
        }
    }
}
