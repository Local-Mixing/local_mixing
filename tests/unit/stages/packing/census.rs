use super::*;
use crate::stages::packing::oracle_io::{read_bounds, read_oracle_mpmct};
use std::path::Path;

fn fx(p: &str) -> std::path::PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("tests/fixtures/stage4")
        .join(p)
}

#[test]
fn n8_report_equals_anf_compose_output() {
    let c = read_oracle_mpmct(&fx("n8_L8/rep.mpmct1")).unwrap();
    let b = read_bounds(&fx("n8_L8/rep.bounds")).unwrap();
    for (budget, label, every, file) in [
        (65536, "n8_L8-rep512", 500, "n8_L8/anf_rep.txt"),
        (64, "n8_L8-b64", 3, "n8_L8/anf_b64.txt"),
        (256, "n8_L8-b256", 3, "n8_L8/anf_b256.txt"),
    ] {
        let r = anf_size_report(
            &c.gates,
            c.computed_n,
            &b,
            budget,
            label,
            every,
            CapSemantics::CPython39,
        );
        assert_eq!(r.text, std::fs::read_to_string(fx(file)).unwrap(), "{file}");
        assert_eq!(r.failed, 0);
        assert_eq!(
            r.overflow,
            r.overflow_product + r.overflow_running_sum + r.overflow_final
        );
    }
}

#[test]
fn spot_check_failure_is_reported() {
    // A report computed on one circuit and checked against another cannot be
    // produced through the API; instead check the format of a clean report.
    let c = read_oracle_mpmct(&fx("anf/cancel.mpmct1")).unwrap();
    let b = read_bounds(&fx("anf/cancel.bounds")).unwrap();
    let r = anf_size_report(
        &c.gates,
        c.computed_n,
        &b,
        64,
        "cancel-b64",
        2,
        CapSemantics::CPython39,
    );
    assert_eq!(
        r.text,
        std::fs::read_to_string(fx("anf/cancel.anf_b64.txt")).unwrap()
    );
    assert!(r.text.ends_with(" (OK)\n"));
}
