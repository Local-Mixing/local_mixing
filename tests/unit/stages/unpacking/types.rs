use super::*;
use crate::stages::packing::types::Stage4Error;

#[test]
fn exit_codes_and_meanings() {
    assert_eq!(Stage7Error::Usage("x".into()).exit_code(), 2);
    assert_eq!(Stage7Error::Input("x".into()).exit_code(), 3);
    assert_eq!(Stage7Error::Verify("x".into()).exit_code(), 4);
    assert_eq!(
        Stage7Error::io("p", io::Error::other("boom")).exit_code(),
        7
    );
    assert_eq!(Stage7Error::Internal("x".into()).exit_code(), 8);
    for (c, m) in EXIT_CODES {
        assert_eq!(Stage7Error::meaning(c), m);
    }
    assert_eq!(Stage7Error::meaning(42), "unknown exit code");
    let e = Stage7Error::io("/x/y", io::Error::other("boom"));
    assert!(e.to_string().contains("/x/y") && e.to_string().contains("boom"));
}

#[test]
fn stage4_errors_map_onto_stage7_codes() {
    let f = |e: Stage4Error| Stage7Error::from(e).exit_code();
    assert_eq!(f(Stage4Error::Usage("u".into())), 2);
    assert_eq!(f(Stage4Error::Input("i".into())), 3);
    assert_eq!(f(Stage4Error::Verify("v".into())), 4);
    assert_eq!(f(Stage4Error::io("p", io::Error::other("x"))), 7);
    assert_eq!(f(Stage4Error::Internal("i".into())), 8);
    assert_eq!(f(Stage4Error::OverBudget { windows: vec![] }), 8);
}

#[test]
fn params_validate() {
    assert!(UnpackParams::default().validate().is_ok());
    let mut p = UnpackParams {
        ancilla_pool: 5000,
        ..Default::default()
    };
    assert!(p.validate().is_err());
    p = UnpackParams::default();
    p.equiv_batches = 0;
    assert!(p.validate().is_err());
    p = UnpackParams::default();
    p.synth.restarts = 0;
    assert!(p.validate().is_err());
}
