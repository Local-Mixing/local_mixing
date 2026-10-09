use super::*;

#[test]
fn strict_reader_accepts_oracle_files() {
    let c = parse_oracle_mpmct("mpmct1 4 3\n0 1 2 1 0 3 1\n\n2 0 0 \n5 0 1 0 1\n").unwrap();
    assert_eq!((c.header_n, c.header_m, c.computed_n), (4, 3, 6));
    assert_eq!(c.gates.len(), 3);
    assert!(c.gates[0].comp);
    assert_eq!(c.gates[0].ctrls.as_slice(), &[(1, false), (3, true)]);
    assert!(c.gates[1].ctrls.is_empty());
    let e = parse_oracle_mpmct("mpmct1 4 0\n").unwrap();
    assert_eq!((e.computed_n, e.gates.len()), (4, 0));
}

#[test]
fn strict_reader_rejects_non_canonical_gates() {
    for (bad, why) in [
        ("mpmct1 4 1\n0 0 2 3 1 1 1\n", "controls not ascending"),
        ("mpmct1 4 1\n0 0 2 1 1 1 0\n", "duplicate control"),
        ("mpmct1 4 1\n0 0 1 0 1\n", "control on target"),
        ("mpmct1 4 1\n0 2 1 1 1\n", "comp 2"),
        ("mpmct1 4 1\n0 0 1 1 2\n", "polarity 2"),
        ("mpmct1 4 2\n0 0 1 1 1\n", "header m"),
        ("mpmct1 4 1\n0 0 1 1 1 7\n", "extra token"),
        ("mpmct1 4 1\n+0 0 0\n", "signed number"),
        ("mpmct1 4 1\n0 0 1 1_0 1\n", "underscore"),
        ("mpmct 4 1\n0 0 0\n", "header tag"),
        ("mpmct1 4 1\n70000 0 0\n", "wire above 65535"),
        ("mpmct1 4 1\n0 0\n", "truncated"),
    ] {
        match parse_oracle_mpmct(bad) {
            Err(Stage4Error::Input(_)) => {}
            other => panic!("{why}: accepted or wrong error: {other:?}"),
        }
    }
}

#[test]
fn bounds_and_cuts_follow_the_oracle() {
    let b = parse_bounds("5\n 3 \n\n-2\n5\n0\n9\n12\n").unwrap();
    assert_eq!(b, vec![5, 3, -2, 5, 0, 9, 12]);
    // [0] + sorted(0 < b < m) + [m], no dedup
    assert_eq!(window_cuts(&b, 9), vec![0, 3, 5, 5, 9]);
    assert_eq!(window_cuts(&[], 0), vec![0, 0]);
    assert!(parse_bounds("1.5\n").is_err());
    assert!(parse_bounds("--3\n").is_err());
}

#[test]
fn writers_match_the_oracle_formats() {
    let c = parse_oracle_mpmct("mpmct1 4 2\n0 1 2 1 0 3 1\n2 0 0 \n").unwrap();
    let mut pack = Vec::new();
    write_pack_mpmct(&mut pack, 4, &c.gates).unwrap();
    assert_eq!(
        String::from_utf8(pack).unwrap(),
        "mpmct1 4 2\n0 1 2 1 0 3 1\n2 0 0\n"
    );
    let mut rep = Vec::new();
    write_repair_mpmct(&mut rep, 4, &c.gates).unwrap();
    assert_eq!(
        String::from_utf8(rep).unwrap(),
        "mpmct1 4 2\n0 1 2 1 0 3 1\n2 0 0 \n"
    );
    let mut b = Vec::new();
    write_bounds(&mut b, &[3, 7]).unwrap();
    assert_eq!(b, b"3\n7\n");
    let mut e = Vec::new();
    write_bounds(&mut e, &[]).unwrap();
    assert_eq!(e, b"\n");
}
