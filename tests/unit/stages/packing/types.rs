use super::*;

#[test]
fn exit_codes_and_meanings_are_a_bijection() {
    let errs = [
        Stage4Error::Usage(String::new()),
        Stage4Error::Input(String::new()),
        Stage4Error::Verify(String::new()),
        Stage4Error::Floor {
            nonoutput_visible: vec![],
        },
        Stage4Error::OverBudget { windows: vec![] },
        Stage4Error::io("x", std::io::Error::other("e")),
        Stage4Error::Internal(String::new()),
    ];
    let codes: Vec<i32> = errs.iter().map(|e| e.exit_code()).collect();
    assert_eq!(codes, vec![2, 3, 4, 5, 6, 7, 8]);
    let mut all: Vec<i32> = EXIT_CODES.iter().map(|(c, _)| *c).collect();
    let mut meanings: Vec<&str> = EXIT_CODES.iter().map(|(_, m)| *m).collect();
    all.dedup();
    meanings.sort_unstable();
    meanings.dedup();
    assert_eq!(all.len(), EXIT_CODES.len());
    assert_eq!(meanings.len(), EXIT_CODES.len());
    for c in codes {
        assert_ne!(Stage4Error::meaning(c), "unknown exit code");
    }
    assert_eq!(Stage4Error::meaning(42), "unknown exit code");
}

#[test]
fn cap_semantics_names_round_trip() {
    for s in [CapSemantics::CPython39, CapSemantics::Canonical] {
        assert_eq!(CapSemantics::parse(s.name()), Some(s));
    }
    assert_eq!(CapSemantics::default(), CapSemantics::CPython39);
    assert_eq!(CapSemantics::parse("sorted"), None);
}
