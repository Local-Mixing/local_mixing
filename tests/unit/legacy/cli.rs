use super::*;

#[test]
fn regular_gate_count_parser_enforces_the_monomial_abi() {
    assert_eq!(parse_regular_gate_count("1"), Ok(1));
    assert_eq!(parse_regular_gate_count("21"), Ok(21));
    assert!(parse_regular_gate_count("0").is_err());
    assert!(parse_regular_gate_count("22").is_err());
    assert!(parse_regular_gate_count("not-a-number").is_err());
}

fn parse_shoot_args(extra: &[&str]) -> Result<clap::ArgMatches, clap::Error> {
    let mut args = vec![
        "sss",
        "--n",
        "3",
        "--m",
        "1",
        "--x",
        "1",
        "--source",
        "source.g57",
        "--rounds",
        "0",
        "--destination",
        "out.mpmct1",
    ];
    args.extend_from_slice(extra);
    add_shoot_args(Command::new("sss")).try_get_matches_from(args)
}

#[test]
fn five_carrier_flag_parses_for_the_cnot_gadgetizer() {
    let matches = parse_shoot_args(&["--cnot", "--gadgetize", "--five-carrier"]).unwrap();
    assert!(matches.get_flag("five_carrier"));
}

#[test]
fn five_carrier_flag_requires_cnot_and_gadgetize() {
    assert!(parse_shoot_args(&["--gadgetize", "--five-carrier"]).is_err());
    assert!(parse_shoot_args(&["--cnot", "--five-carrier"]).is_err());
}

#[test]
fn five_carrier_flag_rejects_the_single_carrier_override() {
    assert!(
        parse_shoot_args(&[
            "--cnot",
            "--gadgetize",
            "--five-carrier",
            "--prod-single",
            "1",
        ])
        .is_err()
    );
}

#[test]
fn strong_five_carrier_flag_parses_and_conflicts_with_legacy_five() {
    let matches = parse_shoot_args(&["--cnot", "--gadgetize", "--strong-five-carrier"]).unwrap();
    assert!(matches.get_flag("strong_five_carrier"));
    assert!(
        parse_shoot_args(&[
            "--cnot",
            "--gadgetize",
            "--five-carrier",
            "--strong-five-carrier",
        ])
        .is_err()
    );
    assert!(
        parse_shoot_args(&[
            "--cnot",
            "--gadgetize",
            "--strong-five-carrier",
            "--prod-single",
            "1",
        ])
        .is_err()
    );
}

#[test]
fn six_carrier_flag_parses_for_the_cnot_gadgetizer() {
    let matches = parse_shoot_args(&["--cnot", "--gadgetize", "--six-carrier"]).unwrap();
    assert!(matches.get_flag("six_carrier"));
}

#[test]
fn six_carrier_flag_requires_cnot_and_gadgetize() {
    assert!(parse_shoot_args(&["--gadgetize", "--six-carrier"]).is_err());
    assert!(parse_shoot_args(&["--cnot", "--six-carrier"]).is_err());
}

#[test]
fn strong_six_carrier_flag_parses_and_conflicts_with_legacy_six() {
    let matches = parse_shoot_args(&["--cnot", "--gadgetize", "--strong-six-carrier"]).unwrap();
    assert!(matches.get_flag("strong_six_carrier"));
    assert!(
        parse_shoot_args(&[
            "--cnot",
            "--gadgetize",
            "--six-carrier",
            "--strong-six-carrier",
        ])
        .is_err()
    );
    assert!(
        parse_shoot_args(&[
            "--cnot",
            "--gadgetize",
            "--strong-six-carrier",
            "--prod-single",
            "1",
        ])
        .is_err()
    );
}

#[test]
fn seven_carrier_flag_parses_for_the_cnot_gadgetizer() {
    let matches = parse_shoot_args(&["--cnot", "--gadgetize", "--seven-carrier"]).unwrap();
    assert!(matches.get_flag("seven_carrier"));
}

#[test]
fn seven_carrier_flag_requires_cnot_and_gadgetize() {
    assert!(parse_shoot_args(&["--gadgetize", "--seven-carrier"]).is_err());
    assert!(parse_shoot_args(&["--cnot", "--seven-carrier"]).is_err());
}

#[test]
fn nonlinear_carrier_flags_are_mutually_exclusive() {
    assert!(
        parse_shoot_args(&["--cnot", "--gadgetize", "--five-carrier", "--six-carrier",]).is_err()
    );
    assert!(
        parse_shoot_args(&[
            "--cnot",
            "--gadgetize",
            "--six-carrier",
            "--prod-single",
            "1",
        ])
        .is_err()
    );
    assert!(
        parse_shoot_args(&["--cnot", "--gadgetize", "--six-carrier", "--seven-carrier",]).is_err()
    );
    assert!(
        parse_shoot_args(&["--cnot", "--gadgetize", "--five-carrier", "--seven-carrier",]).is_err()
    );
    assert!(
        parse_shoot_args(&[
            "--cnot",
            "--gadgetize",
            "--seven-carrier",
            "--prod-single",
            "1",
        ])
        .is_err()
    );
}
